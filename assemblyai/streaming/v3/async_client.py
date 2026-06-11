import asyncio
import collections.abc
import inspect
import json
import logging
from typing import Any, AsyncIterable, Callable, Dict, Iterable, Optional, Union

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


def websocket_connect_async(
    uri: str, additional_headers: Dict[str, Optional[str]]
) -> Any:
    """Open a websocket connection using whichever ``websockets`` API is
    available. Returns the underlying ``Connect`` awaitable so callers may
    ``await`` it directly (or wrap in ``asyncio.wait_for``). Module-level
    indirection so tests can patch a single attribute.

    ``additional_headers`` matches the ``Dict[str, Optional[str]]`` shape
    returned by ``_build_headers``; an ``Authorization`` value of ``None``
    (no credentials configured) is forwarded to the underlying websockets
    library so the misconfiguration surfaces at the handshake layer.
    """
    return _ws_connect(uri, **{_WS_HEADER_KW: additional_headers})


class AsyncStreamingClient(_BaseStreamingClient):
    """Asyncio-native counterpart to ``StreamingClient``.

    The public API mirrors the thread-based client one-to-one — same options,
    parameters, events, and event-handler registration. Methods that touch the
    network are coroutines. Event handlers may be plain callables or
    coroutine functions; coroutine handlers are awaited inline by the single
    internal read task. Handlers should therefore avoid indefinite blocking,
    just as with the sync client.

    Behavioral notes vs. the sync ``StreamingClient``:

    - ``stream`` / ``set_params`` / ``force_endpoint`` raise ``RuntimeError``
      when called before ``connect()`` — silent drop would diverge from the
      sync client (which buffers pre-connect data) in a way that's easy to
      miss. After the connection has closed, the same calls are silent
      no-ops so cleanup paths don't need defensive try/except.
    - ``disconnect(terminate=True)`` waits at most 2.0s for the write task to
      drain the ``TerminateSession`` frame before forcing teardown. The sync
      client joins indefinitely.
    - Supports ``async with``: ``disconnect()`` is invoked on block exit so
      the websocket / HTTP client are always released even when user code
      raises.
    """

    def __init__(self, options: StreamingClientOptions):
        super().__init__(options)

        self._client = _AsyncHTTPClient(
            api_host=options.api_host, api_key=options.api_key
        )

        # Created lazily in ``connect()`` so they bind to the loop that runs
        # ``connect()``, not whatever loop was current at ``__init__`` time
        # (matters on Python 3.8/3.9 and avoids "no running event loop"
        # DeprecationWarnings on 3.10+ when constructed outside a loop).
        self._write_queue: Optional["asyncio.Queue[OperationMessage]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None

    async def connect(self, params: StreamingParameters) -> None:
        # Single-use: a client whose connection went down (success or
        # handshake failure) sets ``_connection_closed_reported``; reusing
        # it would yield a silently dead read/write loop because
        # ``_stop_event`` is already set.
        already_used = (
            self._websocket is not None
            or self._connection_closed_reported
            or (self._read_task is not None and not self._read_task.done())
        )
        if already_used:
            raise RuntimeError(
                "AsyncStreamingClient has already been connected; "
                "create a new instance for a new connection."
            )

        self._write_queue = asyncio.Queue()
        self._stop_event = asyncio.Event()

        _emit_param_warnings(params)

        uri = _build_uri(self._options.api_host, params)
        headers = _build_headers(self._options)
        options = self._options

        for attempt in range(options.max_connection_retries + 1):
            try:
                self._websocket = await asyncio.wait_for(
                    websocket_connect_async(uri, additional_headers=headers),
                    timeout=options.connect_timeout,
                )
                break
            except websockets.exceptions.InvalidStatus as exc:
                # HTTP-level rejection (auth, quota, bad request): a retry would
                # hit the same response, so fail fast.
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                await self._report_connection_closed(
                    StreamingError(
                        message=f"WebSocket handshake rejected (HTTP {status_code})",
                        code=status_code,
                    )
                )
                # Single-use design: a failed handshake terminates the client.
                # Close the HTTP client now so users who treat ``on_error`` as
                # the terminal signal don't leak the httpx pool.
                await self._client.aclose()
                return
            except (
                websockets.exceptions.InvalidHandshake,
                websockets.exceptions.ConnectionClosed,
                OSError,
                asyncio.TimeoutError,
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
                        await asyncio.sleep(options.connection_retry_delay)
                    continue
                await self._report_connection_closed(exc)
                await self._client.aclose()
                return

        self._read_task = asyncio.create_task(
            self._read_loop(), name="AsyncStreamingClient._read_loop"
        )
        self._write_task = asyncio.create_task(
            self._write_loop(), name="AsyncStreamingClient._write_loop"
        )

        logger.debug("Connected to WebSocket server")

    async def disconnect(self, terminate: bool = False) -> None:
        if self._stop_event is None:
            # Never connected — still close the HTTP client so the pool
            # doesn't leak.
            await self._client.aclose()
            return

        # Enqueue Terminate even when stop is already set: ``_write_loop``
        # bypasses the stop gate for TerminateSession so the frame still
        # reaches the server when the write task is alive.
        if terminate and self._write_queue is not None:
            await self._write_queue.put(TerminateSession())
            # Let the write task drain TerminateSession and exit naturally
            # before we set stop / cancel below. ``asyncio.wait`` does not
            # cancel the awaited task on timeout, unlike ``wait_for``.
            if self._write_task is not None and not self._write_task.done():
                await asyncio.wait({self._write_task}, timeout=2.0)
            # Don't stop the read task yet — the server sends the final Turn
            # and TerminationEvent after receiving Terminate. Every terminal
            # path sets ``_stop_event``, so waiting on it here lets those
            # messages dispatch before teardown.
            if (
                self._read_task is not None
                and not self._read_task.done()
                and asyncio.current_task() is not self._read_task
            ):
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._options.terminate_timeout,
                    )
                except asyncio.TimeoutError:
                    pass

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
        # Loud on misuse (pre-connect), quiet on natural close (post-stop).
        # The first guards against silent data loss; the second keeps cleanup
        # paths simple.
        write_queue, stop_event = self._ensure_connected("stream")
        if stop_event.is_set():
            return

        if isinstance(data, bytes):
            await write_queue.put(data)
            return

        if isinstance(data, collections.abc.AsyncIterable):
            async for chunk in data:
                if stop_event.is_set():
                    return
                await write_queue.put(chunk)
            return

        for chunk in data:
            if stop_event.is_set():
                return
            await write_queue.put(chunk)

    async def set_params(self, params: StreamingSessionParameters) -> None:
        write_queue, stop_event = self._ensure_connected("set_params")
        if stop_event.is_set():
            return
        message_dict = _normalize_min_turn_silence(_dump_model(params))
        message = UpdateConfiguration(**message_dict)
        await write_queue.put(message)

    async def force_endpoint(self) -> None:
        write_queue, stop_event = self._ensure_connected("force_endpoint")
        if stop_event.is_set():
            return
        await write_queue.put(ForceEndpoint())

    def _ensure_connected(
        self, method: str
    ) -> "tuple[asyncio.Queue[OperationMessage], asyncio.Event]":
        # Returns the post-connect primitives so callers narrow ``Optional``
        # locally instead of repeating ``is None`` checks at every use site
        # (mypy can't propagate narrowing through a separate method call).
        if self._write_queue is None or self._stop_event is None:
            raise RuntimeError(
                f"AsyncStreamingClient is not connected; call connect() before {method}()"
            )
        return self._write_queue, self._stop_event

    async def _write_loop(self) -> None:
        # ``_write_loop`` is only ``create_task``ed inside ``connect()`` after
        # the primitives are initialized. ``if`` (not ``assert``) so it
        # survives ``python -O`` if the invariant is ever violated.
        if self._write_queue is None or self._stop_event is None:
            raise RuntimeError("AsyncStreamingClient internal state not initialized")
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
        # ``_read_loop`` is only ``create_task``ed inside ``connect()`` after
        # ``_stop_event`` is initialized. ``if`` (not ``assert``) so it
        # survives ``python -O`` if the invariant is ever violated.
        if self._stop_event is None:
            raise RuntimeError("AsyncStreamingClient internal state not initialized")
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
        # ``_handle_message`` is only reached from ``_read_loop``, which only
        # runs after ``connect()`` has initialized ``_stop_event``.
        if self._stop_event is None:
            raise RuntimeError("AsyncStreamingClient internal state not initialized")
        if isinstance(message, TerminationEvent):
            self._stop_event.set()

        event_type = StreamingEvents[message.type]

        for handler in self._handlers[event_type]:
            await self._invoke_handler(handler, message, event_type)

    async def _handle_warning(self, warning: WarningEvent) -> None:
        logger.warning(
            "Streaming warning (code=%s): %s", warning.warning_code, warning.warning
        )
        for handler in self._handlers[StreamingEvents.Warning]:
            await self._invoke_handler(handler, warning, StreamingEvents.Warning)

    async def _report_server_error(self, error: ErrorEvent) -> None:
        # Only reachable from ``_read_loop`` (after primitives are initialized).
        if self._stop_event is None:
            raise RuntimeError("AsyncStreamingClient internal state not initialized")
        self._server_error_reported = True
        streaming_error = StreamingError(message=error.error, code=error.error_code)
        logger.error("Streaming error: %s (code=%s)", error.error, error.error_code)
        await self._dispatch_error(streaming_error)
        # Tear down locally so a server that sends Error without a trailing
        # close frame doesn't leave the read loop blocked in ``recv()``
        # forever. ``_close_websocket`` is idempotent; if the trailing close
        # does arrive, ``_report_connection_closed`` will dedup via
        # ``_server_error_reported``.
        await self._close_websocket()
        self._stop_event.set()

    async def _report_connection_closed(
        self,
        error: Union[
            StreamingError,
            ErrorEvent,
            websockets.exceptions.ConnectionClosed,
            OSError,
        ],
    ) -> None:
        # Callers (``connect()`` failure path, ``_read_loop``, ``_write_loop``)
        # all run after ``_stop_event`` is initialized.
        if self._stop_event is None:
            raise RuntimeError("AsyncStreamingClient internal state not initialized")
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
            await self._invoke_handler(handler, error, StreamingEvents.Error)

    async def _invoke_handler(
        self,
        handler: Callable,
        payload: Any,
        event_type: StreamingEvents,
    ) -> None:
        try:
            result = handler(self, payload)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("on_%s handler raised", event_type.name.lower())

    async def create_temporary_token(
        self,
        expires_in_seconds: int,
        max_session_duration_seconds: Optional[int] = None,
    ) -> str:
        return await self._client.create_temporary_token(
            expires_in_seconds=expires_in_seconds,
            max_session_duration_seconds=max_session_duration_seconds,
        )

    async def __aenter__(self) -> "AsyncStreamingClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect(terminate=exc_type is None)


class _AsyncHTTPClient:
    def __init__(self, api_host: str, api_key: Optional[str] = None):
        # Lazy: don't instantiate httpx.AsyncClient here. Bare construction of
        # an AsyncStreamingClient that's never connected (or used only for
        # connect() — which doesn't go through the HTTP client) must not
        # leak an httpx pool.
        self._api_host = api_host
        self._api_key = api_key
        self._http_client: Optional[httpx.AsyncClient] = None
        self._closed = False

    def _get_or_create_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            headers = {"User-Agent": f"{httpx._client.USER_AGENT} {_user_agent()}"}
            if self._api_key:
                headers["Authorization"] = self._api_key
            self._http_client = httpx.AsyncClient(
                base_url="https://" + self._api_host,
                headers=headers,
            )
        return self._http_client

    async def create_temporary_token(
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

        response = await self._get_or_create_client().get("/v3/token", params=params)
        response.raise_for_status()
        return response.json()["token"]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._http_client is None:
            return
        try:
            await self._http_client.aclose()
        except Exception as exc:
            logger.debug("Error closing async HTTP client: %s", exc)
