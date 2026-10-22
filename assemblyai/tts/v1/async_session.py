import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from ... import _audio
from ... import async_client as _async_client
from .._base_ws import PULL_QUEUE_FRAMES, build_uri, check_text_frame, resolve_settings
from .models import TTSConfig, TTSError
from .ws_models import (
    TTSBeginEvent,
    TTSEvents,
    TTSTerminationEvent,
    error_from_close,
    error_from_frame,
    parse_event,
)

__all__ = ["AsyncTTSSession"]

logger = logging.getLogger(__name__)

_Handler = Callable[[Any], Any]
_Item = Tuple[str, Any, Any]


def websocket_connect_async(
    uri: str, additional_headers: Dict[str, Optional[str]]
) -> Any:
    """Opens an asyncio websocket; the streaming client's seam, imported
    lazily so `import assemblyai` does not load websockets. Tests patch this."""

    from ...streaming.v3.async_client import websocket_connect_async as _connect

    return _connect(uri, additional_headers=additional_headers)


class AsyncTTSSession:
    """
    A Text-to-Speech WebSocket session for asyncio: the counterpart of
    `TTSSession`.

    Open it with `async with`, which connects and waits for `Begin`.
    `generate()`, `flush()`, `cancel()` and `keep_alive()` are coroutines;
    `speak()` and `audio()` are async iterators of audio bytes. Handlers
    registered with `on()` may be plain functions or coroutines.

    Example:
        ```python
        async with aai.AsyncTTSSession(config=aai.TTSConfig(voice_id="aai-jane")) as session:
            async for chunk in session.speak("Hello there."):
                await play(chunk)
        ```
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        word_boundaries: bool = False,
        inactivity_timeout: Optional[int] = None,
        connect_timeout: float = 10.0,
        terminate_timeout: float = 10.0,
    ) -> None:
        """Creates a session; see `TTSSession.__init__` for the arguments."""

        _audio.check_config(type(self).__name__, config, TTSConfig)
        self.config = config or TTSConfig()
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
        # Created in `connect()`, so they bind to the running loop.
        self._reader: Optional["asyncio.Task[None]"] = None
        self._begun: Any = None
        self._stopped: Any = None
        self._queue: Any = None
        self._next_flush_id = 0
        self._cancel_ids: List[int] = []
        self._cancelled_before = 0
        self._drop_warned = False
        self.begin: Optional[TTSBeginEvent] = None
        self.termination: Optional[TTSTerminationEvent] = None
        self.error: Optional[TTSError] = None

    def on(self, event: Union[TTSEvents, str], handler: _Handler) -> None:
        """Registers `handler(event)`, a function or coroutine function."""

        name = str(event)
        if name not in TTSEvents.__members__:
            raise ValueError(
                f"unknown event {name!r}; one of {', '.join(TTSEvents.__members__)}"
            )
        self._handlers.setdefault(name, []).append(handler)

    async def connect(self) -> TTSBeginEvent:
        """Opens the socket and waits for `Begin`; see `TTSSession.connect`."""

        if self._ws is not None:
            raise RuntimeError("the session is already connected")
        self._begun = asyncio.Event()
        self._stopped = asyncio.Event()
        self._queue = asyncio.Queue(maxsize=PULL_QUEUE_FRAMES)
        self._next_flush_id = 0
        self._cancel_ids = []
        self._cancelled_before = 0
        self._drop_warned = False
        self.begin = None
        self.termination = None
        self.error = None

        uri, headers = build_uri(
            self._settings.tts_base_url,
            self.config,
            token=self._token,
            api_key=self._settings.api_key,
            word_boundaries=self._word_boundaries,
            inactivity_timeout=self._inactivity_timeout,
        )
        try:
            self._ws = await asyncio.wait_for(
                websocket_connect_async(uri, headers), timeout=self._connect_timeout
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise TTSError(
                f"WebSocket handshake failed: {exc}", status_code=status
            ) from exc

        self._reader = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait_for(self._begun.wait(), timeout=self._connect_timeout)
        except asyncio.TimeoutError:
            await self.close()
            raise TTSError("The session did not begin in time") from None
        if self.error is not None:
            await self.close()
            raise self.error
        if self.begin is None:
            await self.close()
            raise TTSError("The session closed before it began")
        return self.begin

    async def terminate(self) -> Optional[TTSTerminationEvent]:
        """Ends the session cleanly; see `TTSSession.terminate`."""

        if self._ws is None:
            return None
        if not self._stopped.is_set():
            try:
                await self._send({"type": "Terminate"})
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._terminate_timeout
                )
            except asyncio.TimeoutError:
                pass
        await self.close()
        return self.termination

    async def close(self) -> None:
        """Closes the socket without `Terminate`."""

        ws, self._ws = self._ws, None
        if self._stopped is not None:
            self._stopped.set()
        if self._begun is not None:
            self._begun.set()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if self._queue is not None:
            self._put(("closed", None, None))
        reader = self._reader
        if reader is not None and reader is not asyncio.current_task():
            try:
                await asyncio.wait_for(reader, timeout=self._terminate_timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()

    async def __aenter__(self) -> "AsyncTTSSession":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, *exc: object) -> None:
        if exc_type is None:
            await self.terminate()
        else:
            await self.close()

    async def generate(self, text: str) -> None:
        """Appends `text` to the current request; see `TTSSession.generate`."""

        check_text_frame(text)
        await self._send({"type": "Generate", "text": text})

    async def flush(self) -> int:
        """Concludes the current request and returns its `flush_id`."""

        flush_id = self._next_flush_id
        self._next_flush_id += 1
        await self._send({"type": "Flush"})
        return flush_id

    async def cancel(self) -> None:
        """Barge-in; see `TTSSession.cancel`."""

        cancel_id = self._next_flush_id
        self._next_flush_id += 1
        self._cancel_ids.append(cancel_id)
        self._cancelled_before = max(self._cancelled_before, cancel_id)
        await self._send({"type": "Cancel"})

    async def keep_alive(self) -> None:
        await self._send({"type": "KeepAlive"})

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        """Speaks one text and yields its audio; see `TTSSession.speak`."""

        await self.generate(text)
        flush_id = await self.flush()
        async for chunk in self.audio(flush_id):
            yield chunk

    async def audio(self, flush_id: Optional[int] = None) -> AsyncIterator[bytes]:
        """
        Yields the audio of request `flush_id`; see `TTSSession.audio`.

        Without `flush_id`, the request being written now, so a task can
        play it while another still sends text.
        """

        if self._queue is None:
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
        while True:
            if flush_id < self._cancelled_before:  # dropped by a Cancel
                return
            try:
                kind, ident, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                if self._stopped.is_set():
                    if self.error is not None:
                        raise self.error
                    return
                continue
            if kind == "audio":
                if ident == flush_id:
                    yield payload
            elif kind == "done":
                if ident == flush_id:
                    return
            elif kind == "cancelled":
                if ident > flush_id:
                    return
            elif kind == "error":
                raise payload
            elif kind == "closed":
                if self.error is not None:
                    raise self.error
                return

    def _put(self, item: _Item) -> None:
        """Queues an item for `audio()`, dropping the oldest audio when full."""

        while True:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    dropped = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                if dropped[0] == "audio" and not self._drop_warned:
                    self._drop_warned = True
                    logger.warning(
                        "AsyncTTSSession: dropping audio nobody reads; iterate "
                        "audio() or register an Audio handler"
                    )

    async def _fail(self, error: TTSError) -> None:
        if self.error is None:
            if error.request_id is None and self.begin is not None:
                error.request_id = self.begin.id
            self.error = error
            await self._emit(TTSEvents.Error, error)
        self._put(("error", None, self.error))
        self._stopped.set()
        self._begun.set()

    async def _send(self, frame: Dict[str, Any]) -> None:
        if self._ws is None or self._stopped.is_set():
            if self.error is not None:
                raise self.error
            raise RuntimeError(
                "the session is not open: use `with` / `async with`, or connect()"
            )
        await self._ws.send(json.dumps(frame))

    async def _read_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        ws = self._ws
        while not self._stopped.is_set() and ws is not None:
            try:
                raw = await ws.recv()
            except ConnectionClosed as exc:
                await self._on_closed(exc)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._stopped.is_set():
                    await self._on_closed(exc)
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
                    await self._dispatch(data)
                except Exception as exc:  # a frame this SDK cannot read
                    await self._fail(TTSError(f"Unreadable frame from the API: {exc}"))
                    return

    async def _dispatch(self, data: Dict[str, Any]) -> None:
        kind = data.get("type")
        if kind == "Error":
            await self._fail(error_from_frame(data))
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
            cancel_id = self._cancel_ids.pop(0) if self._cancel_ids else -1
            self._cancelled_before = max(self._cancelled_before, cancel_id)
            self._put(("cancelled", cancel_id, None))
        elif kind == "Termination":
            self.termination = event
        await self._emit(str(kind), event)
        if kind == "Termination":
            self._stopped.set()
            self._put(("closed", None, None))

    async def _on_closed(self, exc: Exception) -> None:
        if self._stopped.is_set():
            return
        # `.code` is 1006 for an abnormal close, where `.rcvd` is None.
        code = getattr(exc, "code", None)
        reason = getattr(exc, "reason", "") or ""
        if code != 1000 and self.error is None:
            error = error_from_close(code, reason)
            if self.begin is not None:
                error.request_id = self.begin.id
            self.error = error
            await self._emit(TTSEvents.Error, self.error)
        self._stopped.set()
        self._begun.set()
        self._put(("closed", None, None))

    async def _emit(self, event: Union[TTSEvents, str], payload: Any) -> None:
        for handler in self._handlers.get(str(event), ()):
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("AsyncTTSSession handler for %s raised", event)
