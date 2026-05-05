import json
import logging
import queue
import sys
import threading
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
from urllib.parse import urlencode

import httpx
import websockets
from pydantic import BaseModel
from websockets.sync.client import connect as websocket_connect

from assemblyai import __version__

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


def _dump_model(model: BaseModel):
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _parse_model(model_class, data):
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(data)
    return model_class.parse_obj(data)


def _normalize_min_turn_silence(params_dict: dict) -> dict:
    """Collapse `min_end_of_turn_silence_when_confident` into `min_turn_silence` so only
    one wire key is ever sent. Emits deprecation warnings."""
    old = params_dict.pop("min_end_of_turn_silence_when_confident", None)
    if old is None:
        return params_dict
    if "min_turn_silence" in params_dict:
        logger.warning(
            "[Deprecation Warning] Both `min_end_of_turn_silence_when_confident` and "
            "`min_turn_silence` are set. Using `min_turn_silence`; "
            "`min_end_of_turn_silence_when_confident` is deprecated."
        )
    else:
        logger.warning(
            "[Deprecation Warning] `min_end_of_turn_silence_when_confident` is "
            "deprecated and will be removed in a future release. Please use "
            "`min_turn_silence` instead."
        )
        params_dict["min_turn_silence"] = old
    return params_dict


def _normalize_voice_focus(params_dict: dict) -> dict:
    """Collapse `noise_suppression_model` / `noise_suppression_threshold` into
    `voice_focus` / `voice_focus_threshold` so only the new wire keys are sent.
    Emits deprecation warnings."""
    for old_key, new_key in (
        ("noise_suppression_model", "voice_focus"),
        ("noise_suppression_threshold", "voice_focus_threshold"),
    ):
        old = params_dict.pop(old_key, None)
        if old is None:
            continue
        if new_key in params_dict:
            logger.warning(
                f"[Deprecation Warning] Both `{old_key}` and `{new_key}` are set. "
                f"Using `{new_key}`; `{old_key}` is deprecated."
            )
        else:
            logger.warning(
                f"[Deprecation Warning] `{old_key}` is deprecated and will be removed "
                f"in a future release. Please use `{new_key}` instead."
            )
            params_dict[new_key] = old
    return params_dict


def _dump_model_json(model: BaseModel):
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(exclude_none=True)
    return model.json(exclude_none=True)


def _user_agent() -> str:
    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    return (
        f"AssemblyAI/1.0 (sdk=Python/{__version__} runtime_env=Python/{python_version})"
    )


class StreamingClient:
    def __init__(self, options: StreamingClientOptions):
        self._options = options

        self._client = _HTTPClient(api_host=options.api_host, api_key=options.api_key)

        self._handlers: Dict[StreamingEvents, List[Callable]] = {}

        for event in StreamingEvents.__members__.values():
            self._handlers[event] = []

        self._write_queue: queue.Queue[OperationMessage] = queue.Queue()
        self._write_thread = threading.Thread(target=self._write_message)
        self._read_thread = threading.Thread(target=self._read_message)
        self._stop_event = threading.Event()
        # Both flags are read and set only on the read thread (or on the main
        # thread before workers start, for handshake errors). Plain bools are
        # sufficient — no cross-thread synchronization is needed.
        self._connection_closed_reported = False
        self._server_error_reported = False
        # Deliberate single-slot shared-memory handoff: the write thread parks
        # a ConnectionClosed here and the read thread drains it. Synchronization
        # is provided by `_stop_event.set()` (write side) + `recv(timeout=1)`
        # (read side), which together give a happens-before within ~1s.
        self._pending_close_error: Optional[Exception] = None
        self._websocket = None

    def connect(self, params: StreamingParameters) -> None:
        if params.speech_model == "u3-pro":
            logger.warning(
                "[Deprecation Warning] The speech model `u3-pro` is deprecated and will be removed in a future release. "
                "Please use `u3-rt-pro` instead."
            )

        if params.customer_support_audio_capture:
            logger.warning(
                "`customer_support_audio_capture=True` will record session audio. "
                "Only enable this when explicitly coordinating with AssemblyAI support."
            )

        params_dict = _normalize_voice_focus(
            _normalize_min_turn_silence(_dump_model(params))
        )

        # JSON-encode list and dict parameters for proper API compatibility (e.g., keyterms_prompt, llm_gateway)
        for key, value in params_dict.items():
            if isinstance(value, list):
                params_dict[key] = json.dumps(value)
            elif isinstance(value, dict):
                params_dict[key] = json.dumps(value)

        params_encoded = urlencode(params_dict)

        host = self._options.api_host
        if host.startswith(("ws://", "wss://")):
            uri = f"{host}/v3/ws?{params_encoded}"
        else:
            uri = f"wss://{host}/v3/ws?{params_encoded}"
        headers = {
            "Authorization": self._options.token
            if self._options.token
            else self._options.api_key,
            "User-Agent": _user_agent(),
            "AssemblyAI-Version": "2025-05-12",
        }

        try:
            self._websocket = websocket_connect(
                uri,
                additional_headers=headers,
                open_timeout=15,
            )
        except websockets.exceptions.InvalidStatus as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
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
            self._report_connection_closed(exc)
            return

        self._write_thread.start()
        self._read_thread.start()

        logger.debug("Connected to WebSocket server")

    def disconnect(self, terminate: bool = False) -> None:
        if terminate and not self._stop_event.is_set():
            self._write_queue.put(TerminateSession())

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

    def on(self, event: StreamingEvents, handler: Callable) -> None:
        if event in StreamingEvents.__members__.values() and callable(handler):
            self._handlers[event].append(handler)

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
                logger.warning(f"Unsupported event type: {message_json['type']}")

    def _handle_message(self, message: EventMessage) -> None:
        if isinstance(message, TerminationEvent):
            self._stop_event.set()

        event_type = StreamingEvents[message.type]

        for handler in self._handlers[event_type]:
            handler(self, message)

    def _parse_message(self, data: Dict[str, Any]) -> Optional[EventMessage]:
        if "type" in data:
            message_type = data.get("type")

            event_type = self._parse_event_type(message_type)

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

    def _handle_warning(self, warning: WarningEvent):
        logger.warning(
            "Streaming warning (code=%s): %s", warning.warning_code, warning.warning
        )
        for handler in self._handlers[StreamingEvents.Warning]:
            handler(self, warning)

    def _report_server_error(self, error: ErrorEvent) -> None:
        self._server_error_reported = True
        streaming_error = StreamingError(
            message=error.error,
            code=error.error_code,
        )
        logger.error("Streaming error: %s (code=%s)", error.error, error.error_code)
        self._dispatch_error(streaming_error)

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
        vi = sys.version_info
        python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
        user_agent = f"{httpx._client.USER_AGENT} AssemblyAI/1.0 (sdk=Python/{__version__} runtime_env=Python/{python_version})"

        headers = {"User-Agent": user_agent}

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
        params: Dict[str, Any] = {}

        if expires_in_seconds:
            params["expires_in_seconds"] = expires_in_seconds

        if max_session_duration_seconds:
            params["max_session_duration_seconds"] = max_session_duration_seconds

        response = self._http_client.get(
            "/v3/token",
            params=params,
        )

        response.raise_for_status()
        return response.json()["token"]
