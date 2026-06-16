import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Generator, Iterable, Optional, Union

import httpx
import websockets
from pydantic import BaseModel
from websockets.sync.client import connect as websocket_connect

from ._base import (
    _BaseStreamingClient,
    _build_headers,
    _build_uri,
    _dump_model,
    _dump_model_json,
    _emit_param_warnings,
    _normalize_min_turn_silence,
    _user_agent,
)
from .models import (
    ErrorEvent,
    EventMessage,
    ForceEndpoint,
    KeepAlive,
    OperationMessage,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminateSession,
    TerminationEvent,
    UpdateConfiguration,
    WarningEvent,
)

logger = logging.getLogger(__name__)


class StreamingClient(_BaseStreamingClient):
    def __init__(self, options: StreamingClientOptions):
        super().__init__(options)

        self._client = _HTTPClient(api_host=options.api_host, api_key=options.api_key)

        self._write_queue: queue.Queue[OperationMessage] = queue.Queue()
        self._write_thread = threading.Thread(target=self._write_message)
        self._read_thread = threading.Thread(target=self._read_message)
        self._stop_event = threading.Event()
        # Deliberate single-slot shared-memory handoff: the write thread parks
        # a ConnectionClosed here and the read thread drains it. Synchronization
        # is provided by `_stop_event.set()` (write side) + `recv(timeout=1)`
        # (read side), which together give a happens-before within ~1s.
        self._pending_close_error: Optional[Exception] = None

    def connect(self, params: StreamingParameters) -> None:
        """Open the WebSocket session and start the read/write threads.

        Blocks until the handshake completes. A transient handshake failure
        (timeout, network drop) is retried up to
        ``options.max_connection_retries`` times before the failure is
        reported. If the server rejects the handshake at the HTTP layer (auth
        error, etc.) ``Error`` is dispatched to any
        ``on(StreamingEvents.Error, ...)`` handler rather than raised, so
        registration order matters: call ``on()`` before ``connect()``.
        """
        _emit_param_warnings(params)

        uri = _build_uri(self._options.api_host, params)
        headers = _build_headers(self._options)
        options = self._options

        for attempt in range(options.max_connection_retries + 1):
            try:
                self._websocket = websocket_connect(
                    uri,
                    additional_headers=headers,
                    open_timeout=options.connect_timeout,
                )
                break
            except websockets.exceptions.InvalidStatus as exc:
                # HTTP-level rejection (auth, quota, bad request): a retry
                # would hit the same response, so fail fast.
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                self._report_connection_closed(
                    StreamingError(
                        message=f"WebSocket handshake rejected (HTTP {status_code})",
                        code=status_code,
                    )
                )
                return
            except (
                websockets.exceptions.InvalidHandshake,
                websockets.exceptions.ConnectionClosed,
                OSError,
                TimeoutError,
            ) as exc:
                if attempt < options.max_connection_retries:
                    logger.debug(
                        "WebSocket connect attempt %d/%d failed (%s); retrying",
                        attempt + 1,
                        options.max_connection_retries + 1,
                        exc,
                    )
                    if options.connection_retry_delay > 0:
                        time.sleep(options.connection_retry_delay)
                    continue
                self._report_connection_closed(exc)
                return

        self._write_thread.start()
        self._read_thread.start()

        logger.debug("Connected to WebSocket server")

    def disconnect(self, terminate: bool = False) -> None:
        """Stop the read/write threads and close the WebSocket.

        Pass ``terminate=True`` for a graceful close — the client sends a
        ``TerminateSession`` frame and waits up to ``options.terminate_timeout``
        seconds for the server's ``TerminationEvent`` (which reports total
        audio duration). Without ``terminate=True`` the WebSocket is closed
        without notifying the server.
        """
        # Enqueue Terminate even when stop is already set: `_write_message`
        # bypasses the stop gate for TerminateSession so the frame still
        # reaches the server when the write thread is alive.
        if terminate:
            self._write_queue.put(TerminateSession())
            # Don't stop the read thread yet — the server sends the final Turn
            # and TerminationEvent after receiving Terminate. Every terminal
            # path sets `_stop_event` (TerminationEvent via `_handle_message`,
            # server close, server error), so waiting on it here lets those
            # messages dispatch before teardown.
            if (
                self._read_thread.is_alive()
                and threading.current_thread() is not self._read_thread
            ):
                self._stop_event.wait(timeout=self._options.terminate_timeout)

        self._stop_event.set()

        current = threading.current_thread()
        for thread in (self._read_thread, self._write_thread):
            if thread is current or not thread.is_alive():
                continue
            try:
                thread.join()
            except RuntimeError as exc:
                logger.debug("Thread join skipped: %s", exc)

        self._close_websocket()

    def _close_websocket(self) -> None:
        if not self._websocket:
            return
        try:
            self._websocket.close()
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            logger.debug("Error closing websocket: %s", exc)

    def stream(
        self, data: Union[bytes, Generator[bytes, None, None], Iterable[bytes]]
    ) -> None:
        """Send audio bytes to the server.

        Accepts a raw ``bytes`` buffer or any (sync) iterable of ``bytes``.
        Returns once all chunks are enqueued — the write thread does the
        actual sending. After ``disconnect()`` (or a connection drop) this
        becomes a silent no-op.
        """
        if self._stop_event.is_set():
            return

        if isinstance(data, bytes):
            self._write_queue.put(data)
            return

        for chunk in data:
            if self._stop_event.is_set():
                return
            self._write_queue.put(chunk)

    def set_params(self, params: StreamingSessionParameters):
        message_dict = _normalize_min_turn_silence(_dump_model(params))
        message = UpdateConfiguration(**message_dict)
        self._write_queue.put(message)

    def force_endpoint(self):
        message = ForceEndpoint()
        self._write_queue.put(message)

    def keep_alive(self):
        message = KeepAlive()
        self._write_queue.put(message)

    def _write_message(self) -> None:
        while True:
            if not self._websocket:
                raise ValueError("Not connected to the WebSocket server")

            try:
                data = self._write_queue.get(timeout=1)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue

            # TerminateSession bypasses the stop gate so disconnect(terminate=True)
            # can always send it, even when stop is set between put() and the
            # write loop's next iteration.
            is_terminate = isinstance(data, TerminateSession)
            if not is_terminate and self._stop_event.is_set():
                return

            try:
                if isinstance(data, bytes):
                    self._websocket.send(data)
                elif isinstance(data, BaseModel):
                    self._websocket.send(_dump_model_json(data))
                else:
                    raise ValueError(f"Attempted to send invalid message: {type(data)}")
            except websockets.exceptions.ConnectionClosed as exc:
                # Defer reporting to the read thread so all on_error dispatch
                # happens on a single thread (no cross-thread dedup race).
                self._pending_close_error = exc
                self._stop_event.set()
                return

            if is_terminate:
                return

    def _read_message(self) -> None:
        while True:
            if not self._websocket:
                raise ValueError("Not connected to the WebSocket server")

            # Drain a write-thread close before honoring stop, so a stop set by
            # the write thread doesn't cause us to exit silently with an
            # unreported close.
            if self._pending_close_error is not None:
                pending, self._pending_close_error = self._pending_close_error, None
                self._report_connection_closed(pending)
                return
            if self._stop_event.is_set():
                return

            try:
                message_data = self._websocket.recv(timeout=1)
            except TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed as exc:
                self._report_connection_closed(exc)
                return

            try:
                message_json = json.loads(message_data)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to decode message: {exc}")
                continue

            message = self._parse_message(message_json)

            if isinstance(message, ErrorEvent):
                self._report_server_error(message)
            elif isinstance(message, WarningEvent):
                self._handle_warning(message)
            elif message:
                self._handle_message(message)
            else:
                logger.warning(f"Unsupported event type: {message_json.get('type')}")

    def _handle_message(self, message: EventMessage) -> None:
        if isinstance(message, TerminationEvent):
            self._stop_event.set()

        event_type = StreamingEvents[message.type]

        for handler in self._handlers[event_type]:
            try:
                handler(self, message)
            except Exception:
                logger.exception("on_%s handler raised", event_type.name.lower())

    def _handle_warning(self, warning: WarningEvent):
        logger.warning(
            "Streaming warning (code=%s): %s", warning.warning_code, warning.warning
        )
        for handler in self._handlers[StreamingEvents.Warning]:
            try:
                handler(self, warning)
            except Exception:
                logger.exception("on_warning handler raised")

    def _report_server_error(self, error: ErrorEvent) -> None:
        self._server_error_reported = True
        streaming_error = StreamingError(
            message=error.error,
            code=error.error_code,
        )
        logger.error("Streaming error: %s (code=%s)", error.error, error.error_code)
        self._dispatch_error(streaming_error)
        # Tear down locally so a server that sends Error without a trailing
        # close frame doesn't leave the read loop spinning in recv(timeout=1)
        # forever. `_close_websocket` is idempotent; if the trailing close
        # does arrive, `_report_connection_closed` will dedup via
        # `_server_error_reported`.
        self._close_websocket()
        self._stop_event.set()

    def _report_connection_closed(
        self,
        error: Union[
            StreamingError,
            ErrorEvent,
            websockets.exceptions.ConnectionClosed,
            OSError,
        ],
    ) -> None:
        # Idempotent: defensive guard in case future callers (e.g. another
        # connect-time error path) reach this method twice.
        if self._connection_closed_reported:
            return
        self._connection_closed_reported = True
        self._stop_event.set()

        streaming_error = self._build_connection_closed_error(error)

        # Clean close (code 1000) → no streaming_error, nothing to report.
        if streaming_error is None:
            self._close_websocket()
            return

        if isinstance(error, websockets.exceptions.ConnectionClosed):
            reason = error.reason or "no reason given"
            logger.error("Connection closed: %s (code=%s)", reason, error.code)
        else:
            logger.error(
                "Connection failed: %s (code=%s)",
                streaming_error,
                streaming_error.code,
            )

        # If a server Error frame already fired on_error, the close is the
        # effect, not a new cause — log it (above) but skip the duplicate
        # user-visible error.
        if not self._server_error_reported:
            self._dispatch_error(streaming_error)

        self._close_websocket()

    def _dispatch_error(self, error: StreamingError) -> None:
        for handler in self._handlers[StreamingEvents.Error]:
            try:
                handler(self, error)
            except Exception:
                logger.exception("on_error handler raised")

    def create_temporary_token(
        self,
        expires_in_seconds: int,
        max_session_duration_seconds: Optional[int] = None,
    ) -> str:
        return self._client.create_temporary_token(
            expires_in_seconds=expires_in_seconds,
            max_session_duration_seconds=max_session_duration_seconds,
        )


class _HTTPClient:
    def __init__(self, api_host: str, api_key: Optional[str] = None):
        headers = {"User-Agent": f"{httpx._client.USER_AGENT} {_user_agent()}"}

        if api_key:
            headers["Authorization"] = api_key

        self._http_client = httpx.Client(
            base_url="https://" + api_host,
            headers=headers,
        )

    def create_temporary_token(
        self,
        expires_in_seconds: int,
        max_session_duration_seconds: Optional[int] = None,
    ) -> str:
        # ``expires_in_seconds`` is required per the type; always forward it
        # so passing ``0`` reaches the server (where it can be validated)
        # instead of being silently dropped by a falsy check.
        params: Dict[str, Any] = {"expires_in_seconds": expires_in_seconds}

        if max_session_duration_seconds is not None:
            params["max_session_duration_seconds"] = max_session_duration_seconds

        response = self._http_client.get(
            "/v3/token",
            params=params,
        )

        response.raise_for_status()
        return response.json()["token"]
