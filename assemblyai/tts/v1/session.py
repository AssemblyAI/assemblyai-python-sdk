import json
import logging
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from ... import _audio
from ... import client as _client
from .._base_ws import (
    PULL_QUEUE_FRAMES,
    PullBuffer,
    PullItem,
    build_uri,
    check_session_config,
    check_text_frame,
    handshake_error,
    max_generate_chars,
    resolve_settings,
)
from .models import TTSConfig, TTSError
from .ws_models import (
    TTSBeginEvent,
    TTSEvents,
    TTSTerminationEvent,
    close_code_and_reason,
    error_from_close,
    error_from_frame,
    parse_event,
)

__all__ = ["TTSSession"]

logger = logging.getLogger(__name__)

_Handler = Callable[[Any], None]


def websocket_connect(uri: str, **kwargs: Any) -> Any:
    """Opens a sync websocket; the streaming client's seam, imported lazily so
    `import assemblyai` does not load websockets. Tests patch this name."""

    import inspect

    from websockets.sync.client import connect as _ws_connect

    from ...streaming.v3.client import websocket_connect as _connect

    # websockets 17 deprecates a bare connect() unless `legacy=True` (older
    # releases have no such parameter).
    if "legacy" in inspect.signature(_ws_connect).parameters:
        kwargs["legacy"] = True
    return _connect(uri, **kwargs)


class TTSSession:
    """
    A Text-to-Speech WebSocket session: text in as it is written, audio out.

    Open it with `with`, which connects and waits for `Begin`. Send text with
    `generate()` as it is produced, conclude each request with `flush()`, and
    read its audio with `audio()`; `speak()` does all three for one text.
    `cancel()` is the barge-in: it stops the request in flight and drops
    whatever is queued. Leaving the `with` block sends `Terminate` and waits
    for the session's `Termination` summary.

    Audio is delivered one of two ways: pulled with `speak()` / `audio()`, or
    pushed to a handler registered with `on(TTSEvents.Audio, ...)`. Register
    handlers for the other events (`FlushDone`, `WordBoundaries`,
    `Cancelled`, `Termination`, `Error`) as needed; a handler takes the event
    and runs on the session's reader thread. A handler may send
    (`generate()`, `flush()`, `cancel()`, `keep_alive()`, `close()`) but not
    wait on the session: `speak()`, `audio()` and `terminate()` raise
    `RuntimeError` there.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        with aai.TTSSession(config=aai.TTSConfig(voice_id="aai-jane")) as session:
            for sentence in reply_sentences():
                for chunk in session.speak(sentence):
                    play(chunk)  # 24 kHz s16le mono
        ```
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        *,
        client: Optional[_client.Client] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        word_boundaries: bool = False,
        inactivity_timeout: Optional[int] = None,
        connect_timeout: float = 10.0,
        terminate_timeout: float = 10.0,
    ) -> None:
        """
        Creates a session; nothing connects until `with` or `connect()`.

        Args:
            config: The voice and audio options (`voice_id`, `model_id`,
                `language`, `sample_rate`, `encoding`). A session streams raw
                PCM or G.711: `container` may only be `raw`, and `bit_rate`
                does not apply.
            client: A `Client` whose settings (API key, `tts_base_url`) to
                use. Defaults to the shared default client.
            api_key: The API key to authenticate with; takes precedence over
                `client` and the global settings.
            token: A temporary token instead of an API key, for a session
                opened on a user's device. The API spends the token when the
                session begins, so such a session connects once: to
                reconnect, mint a new token and build a new session.
            word_boundaries: Whether the server sends `WordBoundaries` frames.
                Off by default here (the API's own default is on); pass
                `True` to receive them.
            inactivity_timeout: Seconds (5 to 120) with no frame either way
                after which the API closes the session; `None` keeps the
                API's 120 s.
            connect_timeout: Seconds to wait for the handshake and `Begin`.
            terminate_timeout: Seconds to wait for `Termination` on exit.

        Raises:
            TypeError: if `config` is not a `TTSConfig`.
            ValueError: if `config` sets a `wav` container or `bit_rate`.
        """
        _audio.check_config(type(self).__name__, config, TTSConfig)
        self.config = config or TTSConfig()
        check_session_config(self.config)
        self._settings = resolve_settings(
            type(self).__name__, client.settings if client else None, api_key, token
        )
        self._token = token
        self._word_boundaries = word_boundaries
        self._inactivity_timeout = inactivity_timeout
        self._connect_timeout = connect_timeout
        self._terminate_timeout = terminate_timeout

        self._handlers: Dict[str, List[_Handler]] = {}
        self._ws: Any = None
        self._reader: Optional[threading.Thread] = None
        # Held while a frame is written. flush() and cancel() hold it while
        # they also take the frame's flush_id, so the ids follow wire order.
        self._send_lock = threading.RLock()
        self.begin: Optional[TTSBeginEvent] = None
        "The `Begin` frame, once connected."
        self.termination: Optional[TTSTerminationEvent] = None
        """The `Termination` summary once the session ended cleanly; `None`
        before that, after a failure, or when the wait for it timed out."""
        self.error: Optional[TTSError] = None
        "The error that ended the session, if one did."
        self._reset()

    def _reset(self) -> None:
        """Fresh per-connection state, so a session object can be reused."""

        self._begun = threading.Event()
        self._stopped = threading.Event()
        self._queue = PullBuffer(PULL_QUEUE_FRAMES)
        self._queue_ready = threading.Condition()
        self._next_flush_id = 0
        self._cancel_ids: List[int] = []
        # Requests up to this flush_id were dropped by a Cancel (-1: none).
        self._cancelled_before = -1
        self._drop_warned = False
        self.begin = None
        self.termination = None
        self.error = None
        self._error_raised = False

    # -- lifecycle -----------------------------------------------------------

    def on(self, event: Union[TTSEvents, str], handler: _Handler) -> None:
        """Registers `handler(event)` for a server frame type."""

        name = str(event)
        if name not in TTSEvents.__members__:
            raise ValueError(
                f"unknown event {name!r}; one of {', '.join(TTSEvents.__members__)}"
            )
        self._handlers.setdefault(name, []).append(handler)

    def connect(self) -> TTSBeginEvent:
        """
        Opens the socket and waits for `Begin`.

        A session whose previous connection has ended (cleanly, by `close()`
        or by the API, such as after an `inactivity_timeout`) connects again;
        one built with `token=` cannot, since its token is already spent.

        Raises:
            TTSError: if the API refuses the session (its `Error` frame), the
                handshake is rejected, or `Begin` does not arrive in time.
            RuntimeError: if the session is still connected, from a session
                handler, or while a handler of the previous connection still
                runs after `connect_timeout`.
        """
        self._check_not_in_handler("connect()")
        if self._ws is not None:
            if not self._stopped.is_set():
                raise RuntimeError("the session is already connected")
            self.close()  # the previous connection has ended
        if self._reader is not None and self._reader.is_alive():
            # Its reader would write into the new connection's state.
            self._reader.join(self._connect_timeout)
            if self._reader.is_alive():
                raise RuntimeError(
                    "a handler of the previous connection is still running"
                )
        self._reset()

        uri, headers = build_uri(
            self._settings.tts_base_url,
            self.config,
            token=self._token,
            api_key=self._settings.api_key,
            word_boundaries=self._word_boundaries,
            inactivity_timeout=self._inactivity_timeout,
        )
        try:
            self._ws = websocket_connect(
                uri, additional_headers=headers, open_timeout=self._connect_timeout
            )
        except Exception as exc:  # the handshake failed
            self.close()  # ends any audio() waiting on the fresh state
            raise handshake_error(exc) from exc

        self._reader = threading.Thread(
            target=self._read_loop, name="assemblyai-tts-session", daemon=True
        )
        self._reader.start()

        if not self._begun.wait(self._connect_timeout):
            self.close()
            raise TTSError("The session did not begin in time")
        if self.error is not None:
            self.close()
            raise self._surfaced()
        if self.begin is None:
            self.close()
            raise TTSError("The session closed before it began")
        return self.begin

    def terminate(self) -> Optional[TTSTerminationEvent]:
        """
        Ends the session cleanly: `Terminate`, then wait for `Termination`.

        Text sent and not flushed is spoken first. Returns the summary, or
        `None` if the session had already ended or the wait timed out.

        Raises:
            RuntimeError: from a session handler.
        """
        self._check_not_in_handler("terminate()")
        if self._ws is None:
            return None
        if not self._stopped.is_set():
            try:
                self._send({"type": "Terminate"})
            except Exception:  # already closed
                pass
            self._stopped.wait(self._terminate_timeout)
        self.close()
        return self.termination

    def close(self) -> None:
        """Closes the socket without `Terminate`; in-flight audio is dropped."""

        ws, self._ws = self._ws, None
        self._stopped.set()
        self._begun.set()
        if ws is not None:
            try:
                ws.close()
            except Exception:  # already closed
                pass
        self._put(("closed", None, None))
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=self._terminate_timeout)

    def __enter__(self) -> "TTSSession":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, *exc: object) -> None:
        if exc_type is not None:
            self.close()
            return
        self.terminate()
        # A failure nothing in the block observed (push mode, or text sent
        # and never read) must not pass silently.
        if self.error is not None and not self._error_raised:
            raise self._surfaced()

    # -- sending ---------------------------------------------------------------

    def generate(self, text: str) -> None:
        """
        Appends `text` to the current request.

        Consecutive texts are joined exactly as sent, so include the spaces.
        Audio may start before `flush()` once enough complete sentences have
        arrived; `flush()` is what concludes the request.

        Raises:
            ValueError: for a text over the limit of one `Generate` frame
                (split it): the limit `Begin` reports, else 2,000 characters.
        """
        check_text_frame(text, max_generate_chars(self.begin))
        self._send({"type": "Generate", "text": text})

    def flush(self) -> int:
        """
        Concludes the current request and returns its `flush_id`.

        Everything sent so far is spoken, then `FlushDone` arrives for this
        id. Flushes can be pipelined; the audio comes in order.
        """
        with self._send_lock:
            flush_id = self._next_flush_id
            self._next_flush_id += 1
            self._send({"type": "Flush"})
        return flush_id

    def cancel(self) -> None:
        """
        Barge-in: stops the request in flight and drops queued requests.

        An `audio()` iteration of a cancelled request ends at once; audio
        of it still in transit is discarded. The server answers `Cancelled`.
        Text sent and not flushed is discarded. Like `flush()`, a cancel
        takes a `flush_id`, so the next request's id skips one.
        """
        with self._send_lock:
            cancel_id = self._next_flush_id
            self._next_flush_id += 1
            self._cancel_ids.append(cancel_id)
            self._cancelled_before = max(self._cancelled_before, cancel_id)
            self._send({"type": "Cancel"})

    def keep_alive(self) -> None:
        """
        Restarts the API's idle timer. Send one at least once per
        `inactivity_timeout` while the session waits, which includes the
        time already received audio takes to play.
        """

        self._send({"type": "KeepAlive"})

    def speak(self, text: str) -> Iterator[bytes]:
        """
        Speaks one text: `generate()`, `flush()`, then its audio.

        Text longer than one `Generate` frame is split across frames; very
        long text (several thousand characters) should be paced instead, as
        the API limits unspoken text. Returns an iterator of the request's
        audio bytes that ends at its `FlushDone`, or early if the request
        is cancelled.

        Raises:
            TTSError: from the iteration, if the session fails.
            RuntimeError: from a session handler.
        """
        self._check_not_in_handler("speak()")
        check_text_frame(text, 0)
        limit = max_generate_chars(self.begin) or max(len(text), 1)
        for start in range(0, len(text), limit):
            self.generate(text[start : start + limit])
        return self.audio(self.flush())

    def audio(self, flush_id: Optional[int] = None) -> Iterator[bytes]:
        """
        Yields the audio of request `flush_id` until its `FlushDone`.

        Without `flush_id`, the request being written now: a thread can read
        it while `generate()` is still sending text, since audio starts
        before `flush()`. Requests are served in order, so read them in
        order. Audio is queued for pulling only while no `TTSEvents.Audio`
        handler is registered.

        Raises:
            TTSError: if the session fails before the request completes.
            ValueError: for a `flush_id` no `flush()` has returned.
            RuntimeError: when an `Audio` handler receives the audio instead,
                before the session first connects, or from a session handler.
        """
        self._check_not_in_handler("audio()")
        if self._reader is None:
            raise RuntimeError("the session is not open")
        if flush_id is None:
            flush_id = self._next_flush_id  # the request being written now
        elif flush_id > self._next_flush_id:
            raise ValueError(f"flush_id {flush_id} has not been flushed yet")
        if TTSEvents.Audio.value in self._handlers:
            raise RuntimeError(
                "audio goes to the registered Audio handler; pull with audio() "
                "or speak() only without one"
            )
        connection = self._queue  # this connection's; a reconnect makes a new one
        while True:
            if self._queue is not connection:
                return
            # A Cancel drops every request before it and the text being
            # written, whose id it takes.
            if flush_id <= self._cancelled_before:
                return
            item = self._get(timeout=0.5)
            if item is None:
                if self._stopped.is_set():
                    if self.error is not None:
                        raise self._surfaced()
                    return
                continue
            kind, ident, payload = item
            if kind == "audio":
                if ident == flush_id:
                    yield payload
            elif kind == "done":
                if ident == flush_id:
                    return
            elif kind == "cancelled":
                if ident >= flush_id:  # a Cancel sent after this request
                    return
            elif kind == "error":
                raise self._surfaced()
            elif kind == "closed":
                if self.error is not None:
                    raise self._surfaced()
                return

    # -- internals ---------------------------------------------------------------

    def _surfaced(self) -> TTSError:
        """The session's error, marked as raised to the caller."""

        if self.error is None:
            raise RuntimeError("the session has no error")
        self._error_raised = True
        return self.error

    def _check_not_in_handler(self, method: str) -> None:
        """Refuses a method that waits on the reader when the reader, running
        a handler, is the caller: it would wait for itself."""

        if self._reader is not None and threading.current_thread() is self._reader:
            raise RuntimeError(
                f"{method} cannot be called from a session handler: it waits "
                "for the session's reader, which runs the handlers"
            )

    def _put(self, item: PullItem) -> None:
        """Queues an item for `audio()`; past the cap the oldest audio drops."""

        with self._queue_ready:
            dropped = self._queue.append(item)
            self._queue_ready.notify_all()
        if dropped and not self._drop_warned:
            self._drop_warned = True
            logger.warning(
                "TTSSession: over %d audio frames are waiting to be read, so the "
                "oldest are being dropped; read audio() as fast as the audio "
                "arrives, or register an Audio handler",
                self._queue.max_audio,
            )

    def _get(self, timeout: float) -> Optional[PullItem]:
        """The next item for `audio()`; `None` when none came in `timeout` s."""

        with self._queue_ready:
            if not self._queue:
                self._queue_ready.wait(timeout)
            return self._queue.popleft()

    def _send(self, frame: Dict[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        with self._send_lock:
            ws = self._ws
            if ws is None or self._stopped.is_set():
                if self.error is not None:
                    raise self._surfaced()
                raise RuntimeError(
                    "the session is not open: use `with` / `async with`, or connect()"
                )
            try:
                ws.send(json.dumps(frame))
                return
            except ConnectionClosed as exc:
                closed = exc
        # Outside the lock: a handler that sends must not block the reader.
        raise self._closed_error(ws, closed) from closed

    def _closed_error(self, ws: Any, exc: Exception) -> Exception:
        """
        What a send on a socket that closed raises: the session's error, once
        the reader has taken in the frames that arrived before the close.
        """

        if threading.current_thread() is not self._reader:
            self._stopped.wait(1)
        if self.error is not None:
            return self.error
        code, reason = close_code_and_reason(exc)
        if self._ws is not ws or code == 1000:  # closed by close() or cleanly
            return RuntimeError("the session is not open")
        return error_from_close(code, reason)

    def _read_loop(self) -> None:
        try:
            self._read_frames()
        except Exception as exc:  # a failure in the reader itself
            self._fail(TTSError(f"The session's reader failed: {exc!r}"))
        finally:
            # However the reader ended, nothing may wait for it any longer.
            if not self._stopped.is_set():
                self._stopped.set()
                self._begun.set()
                self._put(("closed", None, None))

    def _read_frames(self) -> None:
        from websockets.exceptions import ConnectionClosed

        ws = self._ws
        while not self._stopped.is_set() and ws is not None:
            try:
                raw = ws.recv(timeout=1)
            except TimeoutError:
                continue
            except ConnectionClosed as exc:
                self._on_closed(exc)
                return
            except Exception as exc:  # the socket was closed under us
                if not self._stopped.is_set():
                    self._on_closed(exc)
                return
            if isinstance(raw, bytes):
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                logger.warning("ignoring a non-JSON frame")
                continue
            if isinstance(data, dict):
                try:
                    self._dispatch(data)
                except Exception as exc:  # a frame this SDK cannot read
                    self._fail(TTSError(f"Unreadable frame from the API: {exc}"))
                    return

    def _fail(self, error: TTSError) -> None:
        """Ends the session on an error: handlers, waiters, state."""

        first = self.error is None
        if first:
            if error.request_id is None and self.begin is not None:
                error.request_id = self.begin.id
            self.error = error
        # Waiters learn of the failure before any handler runs.
        self._put(("error", None, self.error))
        self._stopped.set()
        self._begun.set()
        if first:
            self._emit(TTSEvents.Error, error)

    def _dispatch(self, data: Dict[str, Any]) -> None:
        kind = data.get("type")
        if kind == "Error":
            self._fail(error_from_frame(data))
            return
        event = parse_event(data)
        if event is None:
            return
        if kind == "Begin":
            self.begin = event
            self._begun.set()
        elif kind == "Audio":
            if TTSEvents.Audio.value not in self._handlers:
                self._put(("audio", event.flush_id, event.audio))
        elif kind == "FlushDone":
            self._put(("done", event.flush_id, None))
        elif kind == "Cancelled":
            # cancel() already raised the watermark for this id.
            cancel_id = self._cancel_ids.pop(0) if self._cancel_ids else -1
            self._put(("cancelled", cancel_id, None))
        elif kind == "Termination":
            self.termination = event
        self._emit(str(kind), event)
        if kind == "Termination":
            self._stopped.set()
            self._put(("closed", None, None))

    def _on_closed(self, exc: Exception) -> None:
        if self._stopped.is_set():
            return
        code, reason = close_code_and_reason(exc)
        error = None
        if code != 1000 and self.error is None:
            error = error_from_close(code, reason)
            if self.begin is not None:
                error.request_id = self.begin.id
            self.error = error
        self._stopped.set()
        self._begun.set()
        self._put(("closed", None, None))
        if error is not None:
            self._emit(TTSEvents.Error, error)

    def _emit(self, event: Union[TTSEvents, str], payload: Any) -> None:
        for handler in self._handlers.get(str(event), ()):
            try:
                handler(payload)
            except Exception:
                logger.exception("TTSSession handler for %s raised", event)
