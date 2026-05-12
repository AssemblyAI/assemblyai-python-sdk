import asyncio
import collections.abc
import inspect
import json
import logging
import sys
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import httpx
import websockets
from pydantic import BaseModel

# Prefer the new asyncio client API (websockets >= 13). Fall back to the legacy
# top-level connect for older versions the SDK still supports per ``setup.py``
# (``websockets>=11.0``). The two APIs differ only in the header-kwarg name
# (``additional_headers`` vs ``extra_headers``); the ``websocket_connect_async``
# wrapper below papers that over so tests and callers see one entry point.
try:
    from websockets.asyncio.client import connect as _ws_connect

    _WS_HEADER_KW = "additional_headers"
except ImportError:  # pragma: no cover - exercised on websockets <13 only
    from websockets.client import connect as _ws_connect  # type: ignore[no-redef]

    _WS_HEADER_KW = "extra_headers"

from assemblyai import __version__

from .client import (
    _build_headers,
    _build_uri,
    _dump_model,
    _dump_model_json,
    _emit_param_warnings,
    _normalize_min_turn_silence,
    _parse_model,
)
from .models import (
    BeginEvent,
    ErrorEvent,
    EventMessage,
    ForceEndpoint,
    LLMGatewayResponseEvent,
    OperationMessage,
    SpeechStartedEvent,
    StreamingClientOptions,
    StreamingError,
    StreamingErrorCodes,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminateSession,
    TerminationEvent,
    TurnEvent,
    UpdateConfiguration,
    WarningEvent,
)

logger = logging.getLogger(__name__)


def websocket_connect_async(uri: str, additional_headers):
    """Open a websocket connection using whichever ``websockets`` API is
    available. Returns the underlying ``Connect`` awaitable so callers may
    ``await`` it directly (or wrap in ``asyncio.wait_for``). Module-level
    indirection so tests can patch a single attribute."""
    return _ws_connect(uri, **{_WS_HEADER_KW: additional_headers})


class AsyncStreamingClient:
    """Asyncio-native counterpart to ``StreamingClient``.

    The public API mirrors the thread-based client one-to-one — same options,
    parameters, events, and event-handler registration. Methods that touch the
    network are coroutines. Event handlers may be plain callables or
    coroutine functions; coroutine handlers are awaited inline by the single
    internal read task. Handlers should therefore avoid indefinite blocking,
    just as with the sync client.
    """

    def __init__(self, options: StreamingClientOptions):
        self._options = options

        self._client = _AsyncHTTPClient(
            api_host=options.api_host, api_key=options.api_key
        )

        self._handlers: Dict[StreamingEvents, List[Callable]] = {}
        for event in StreamingEvents.__members__.values():
            self._handlers[event] = []

        self._write_queue: "asyncio.Queue[OperationMessage]" = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # Dedup flags. Only ``_read_task`` mutates these — the write task on
        # ``ConnectionClosed`` just logs + sets the stop event + exits. Asyncio's
        # ``await ws.recv()`` raises ``ConnectionClosed`` as soon as the socket
        # transitions to closed, so the read task always sees the close
        # naturally — no cross-task error hand-off is required.
        self._connection_closed_reported = False
        self._server_error_reported = False

        self._websocket: Optional[Any] = None
        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None

    async def connect(self, params: StreamingParameters) -> None:
        if self._websocket is not None or (
            self._read_task is not None and not self._read_task.done()
        ):
            raise RuntimeError(
                "AsyncStreamingClient is already connected; "
                "create a new instance for a new connection."
            )

        _emit_param_warnings(params)

        uri = _build_uri(self._options.api_host, params)
        headers = _build_headers(self._options)

        try:
            self._websocket = await asyncio.wait_for(
                websocket_connect_async(uri, additional_headers=headers),
                timeout=15,
            )
        except websockets.exceptions.InvalidStatus as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            await self._report_connection_closed(
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
            asyncio.TimeoutError,
            TimeoutError,
        ) as exc:
            await self._report_connection_closed(exc)
            return

        self._read_task = asyncio.create_task(
            self._read_loop(), name="AsyncStreamingClient._read_loop"
        )
        self._write_task = asyncio.create_task(
            self._write_loop(), name="AsyncStreamingClient._write_loop"
        )

        logger.debug("Connected to WebSocket server")

    async def disconnect(self, terminate: bool = False) -> None:
        if terminate and not self._stop_event.is_set():
            await self._write_queue.put(TerminateSession())
            # Let the write task drain TerminateSession and exit naturally
            # before we set stop / cancel below. ``asyncio.wait`` does not
            # cancel the awaited task on timeout, unlike ``wait_for``.
            if self._write_task is not None and not self._write_task.done():
                await asyncio.wait({self._write_task}, timeout=2.0)

        self._stop_event.set()

        current = asyncio.current_task()
        for task in (self._read_task, self._write_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Streaming task raised during disconnect")

        await self._close_websocket()
        await self._client.aclose()

    async def _close_websocket(self) -> None:
        if not self._websocket:
            return
        try:
            await self._websocket.close()
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            logger.debug("Error closing websocket: %s", exc)

    async def stream(
        self,
        data: Union[bytes, AsyncIterable[bytes], Iterable[bytes]],
    ) -> None:
        if self._stop_event.is_set():
            return

        if isinstance(data, bytes):
            await self._write_queue.put(data)
            return

        if isinstance(data, collections.abc.AsyncIterable):
            async for chunk in data:
                if self._stop_event.is_set():
                    return
                await self._write_queue.put(chunk)
            return

        for chunk in data:
            if self._stop_event.is_set():
                return
            await self._write_queue.put(chunk)

    async def set_params(self, params: StreamingSessionParameters) -> None:
        message_dict = _normalize_min_turn_silence(_dump_model(params))
        message = UpdateConfiguration(**message_dict)
        await self._write_queue.put(message)

    async def force_endpoint(self) -> None:
        await self._write_queue.put(ForceEndpoint())

    def on(self, event: StreamingEvents, handler: Callable) -> None:
        if event in StreamingEvents.__members__.values() and callable(handler):
            self._handlers[event].append(handler)

    async def _write_loop(self) -> None:
        while True:
            if not self._websocket:
                raise ValueError("Not connected to the WebSocket server")

            try:
                data = await asyncio.wait_for(self._write_queue.get(), timeout=1)
            except asyncio.TimeoutError:
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
                    await self._websocket.send(data)
                elif isinstance(data, BaseModel):
                    await self._websocket.send(_dump_model_json(data))
                else:
                    raise ValueError(f"Attempted to send invalid message: {type(data)}")
            except websockets.exceptions.ConnectionClosed as exc:
                # Dispatch the close directly from the write task. The read
                # task may short-circuit on ``_stop_event`` at the top of its
                # loop (e.g. while a buffered message was processed between
                # ``recv()`` calls) and never observe the close in ``recv()``,
                # so the write task can't rely on it to dispatch.
                # ``_report_connection_closed`` is idempotent — its flag check
                # + set is synchronous (no ``await`` between them), so if the
                # read task also raises ``ConnectionClosed`` it'll be a no-op.
                await self._report_connection_closed(exc)
                return

            if is_terminate:
                return

    async def _read_loop(self) -> None:
        while True:
            if not self._websocket:
                raise ValueError("Not connected to the WebSocket server")

            if self._stop_event.is_set():
                return

            try:
                message_data = await self._websocket.recv()
            except websockets.exceptions.ConnectionClosed as exc:
                await self._report_connection_closed(exc)
                return

            try:
                message_json = json.loads(message_data)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to decode message: {exc}")
                continue

            message = self._parse_message(message_json)

            if isinstance(message, ErrorEvent):
                await self._report_server_error(message)
            elif isinstance(message, WarningEvent):
                await self._handle_warning(message)
            elif message:
                await self._handle_message(message)
            else:
                logger.warning(f"Unsupported event type: {message_json.get('type')}")

    async def _handle_message(self, message: EventMessage) -> None:
        if isinstance(message, TerminationEvent):
            self._stop_event.set()

        event_type = StreamingEvents[message.type]

        for handler in self._handlers[event_type]:
            await self._invoke_handler(handler, message)

    async def _handle_warning(self, warning: WarningEvent) -> None:
        logger.warning(
            "Streaming warning (code=%s): %s", warning.warning_code, warning.warning
        )
        for handler in self._handlers[StreamingEvents.Warning]:
            await self._invoke_handler(handler, warning)

    def _parse_message(self, data: Dict[str, Any]) -> Optional[EventMessage]:
        if "type" in data:
            event_type = self._parse_event_type(data.get("type"))

            if event_type == StreamingEvents.Begin:
                return _parse_model(BeginEvent, data)
            elif event_type == StreamingEvents.Termination:
                return _parse_model(TerminationEvent, data)
            elif event_type == StreamingEvents.Turn:
                return _parse_model(TurnEvent, data)
            elif event_type == StreamingEvents.SpeechStarted:
                return _parse_model(SpeechStartedEvent, data)
            elif event_type == StreamingEvents.LLMGatewayResponse:
                return _parse_model(LLMGatewayResponseEvent, data)
            elif event_type == StreamingEvents.Error:
                return _parse_model(ErrorEvent, data)
            elif event_type == StreamingEvents.Warning:
                return _parse_model(WarningEvent, data)
            else:
                return None
        elif "error" in data:
            return _parse_model(ErrorEvent, data)
        return None

    @staticmethod
    def _parse_event_type(message_type: Optional[Any]) -> Optional[StreamingEvents]:
        if not isinstance(message_type, str):
            return None
        try:
            return StreamingEvents[message_type]
        except KeyError:
            return None

    async def _report_server_error(self, error: ErrorEvent) -> None:
        self._server_error_reported = True
        streaming_error = StreamingError(message=error.error, code=error.error_code)
        logger.error("Streaming error: %s (code=%s)", error.error, error.error_code)
        await self._dispatch_error(streaming_error)

    async def _report_connection_closed(
        self,
        error: Union[
            StreamingError,
            ErrorEvent,
            websockets.exceptions.ConnectionClosed,
            OSError,
        ],
    ) -> None:
        if self._connection_closed_reported:
            return
        self._connection_closed_reported = True
        self._stop_event.set()

        streaming_error = self._build_connection_closed_error(error)

        if streaming_error is None:
            await self._close_websocket()
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
            await self._dispatch_error(streaming_error)

        await self._close_websocket()

    async def _dispatch_error(self, error: StreamingError) -> None:
        for handler in self._handlers[StreamingEvents.Error]:
            try:
                await self._invoke_handler(handler, error)
            except Exception:
                logger.exception("on_error handler raised")

    async def _invoke_handler(self, handler: Callable, payload: Any) -> None:
        try:
            result = handler(self, payload)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Streaming handler raised")

    @staticmethod
    def _build_connection_closed_error(
        error: Union[
            StreamingError,
            ErrorEvent,
            websockets.exceptions.ConnectionClosed,
            OSError,
        ],
    ) -> Optional[StreamingError]:
        if isinstance(error, StreamingError):
            return error
        if isinstance(error, ErrorEvent):
            return StreamingError(message=error.error, code=error.error_code)
        if isinstance(error, websockets.exceptions.ConnectionClosed):
            if error.code == 1000:
                return None
            if error.code is not None and error.code in StreamingErrorCodes:
                message = StreamingErrorCodes[error.code]
            else:
                message = error.reason or f"Connection closed (code={error.code})"
            return StreamingError(message=message, code=error.code)
        return StreamingError(message=f"Connection failed: {error}")

    async def create_temporary_token(
        self,
        expires_in_seconds: int,
        max_session_duration_seconds: Optional[int] = None,
    ) -> str:
        return await self._client.create_temporary_token(
            expires_in_seconds=expires_in_seconds,
            max_session_duration_seconds=max_session_duration_seconds,
        )


class _AsyncHTTPClient:
    def __init__(self, api_host: str, api_key: Optional[str] = None):
        vi = sys.version_info
        python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
        user_agent = (
            f"{httpx._client.USER_AGENT} AssemblyAI/1.0 "
            f"(sdk=Python/{__version__} runtime_env=Python/{python_version})"
        )

        headers = {"User-Agent": user_agent}

        if api_key:
            headers["Authorization"] = api_key

        self._http_client = httpx.AsyncClient(
            base_url="https://" + api_host,
            headers=headers,
        )

    async def create_temporary_token(
        self,
        expires_in_seconds: int,
        max_session_duration_seconds: Optional[int] = None,
    ) -> str:
        params: Dict[str, Any] = {}

        if expires_in_seconds:
            params["expires_in_seconds"] = expires_in_seconds

        if max_session_duration_seconds:
            params["max_session_duration_seconds"] = max_session_duration_seconds

        response = await self._http_client.get("/v3/token", params=params)
        response.raise_for_status()
        return response.json()["token"]

    async def aclose(self) -> None:
        try:
            await self._http_client.aclose()
        except Exception as exc:
            logger.debug("Error closing async HTTP client: %s", exc)
