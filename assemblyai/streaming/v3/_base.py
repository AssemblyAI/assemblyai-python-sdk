"""Sync/async-agnostic core for streaming v3 clients.

Houses the pieces that are *exactly* the same between the threaded
``RealTimeTranscriber`` and the asyncio-based ``AsyncRealTimeTranscriber``:

- Wire-format helpers (``_dump_model``, ``_parse_model``, ``_build_uri``,
  ``_build_headers``, parameter normalization, user-agent construction).
- Inbound message parsing (``_parse_message`` + ``_parse_event_type``).
- Connection-closed error mapping (``_build_connection_closed_error``).
- The ``_BaseStreamingClient`` base class with shared init state and
  the ``on(...)`` handler-registration entrypoint.

Subclasses must implement the I/O loops (``_read_*`` / ``_write_*``) plus
``connect``, ``disconnect``, ``stream``, ``set_params``, ``force_endpoint``,
and ``create_temporary_token``. Sync subclasses use plain methods; async
subclasses use ``async def``. The sync/async return-type divergence is
why those methods aren't ``@abstractmethod`` on this base.
"""

import json
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from urllib.parse import urlencode

import websockets
from pydantic import BaseModel

from assemblyai import __version__

from .models import (
    BeginEvent,
    ErrorEvent,
    EventMessage,
    HeartbeatEvent,
    LLMGatewayResponseEvent,
    RealTimeError,
    RealTimeErrorCodes,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriberOptions,
    SpeakerRevisionEvent,
    SpeechStartedEvent,
    TerminationEvent,
    TurnEvent,
    WarningEvent,
)

logger = logging.getLogger(__name__)


_M = TypeVar("_M", bound=BaseModel)


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _dump_model_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(exclude_none=True)
    return model.json(exclude_none=True)


def _parse_model(model_class: Type[_M], data: Dict[str, Any]) -> _M:
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


def _user_agent() -> str:
    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    return (
        f"AssemblyAI/1.0 (sdk=Python/{__version__} runtime_env=Python/{python_version})"
    )


def _emit_param_warnings(params: RealTimeParameters) -> None:
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
    if params.language_code is not None:
        logger.warning(
            "[Deprecation Warning] `language_code` is deprecated and will be removed in a future release. "
            "Please use `language_codes` instead."
        )


def _build_uri(host: str, params: RealTimeParameters) -> str:
    params_dict = _normalize_voice_focus(
        _normalize_min_turn_silence(_dump_model(params))
    )
    # JSON-encode list and dict parameters for proper API compatibility (e.g.,
    # keyterms_prompt, llm_gateway)
    for key, value in params_dict.items():
        if isinstance(value, list):
            params_dict[key] = json.dumps(value)
        elif isinstance(value, dict):
            params_dict[key] = json.dumps(value)

    params_encoded = urlencode(params_dict)

    if host.startswith(("ws://", "wss://")):
        return f"{host}/v3/ws?{params_encoded}"
    return f"wss://{host}/v3/ws?{params_encoded}"


def _build_headers(options: RealTimeTranscriberOptions) -> Dict[str, Optional[str]]:
    # Matches the pre-refactor sync behavior: ``Authorization`` is left as the
    # raw value (may be ``None`` when neither ``token`` nor ``api_key`` is set,
    # which surfaces the misconfiguration through the websockets/httpx layer).
    return {
        "Authorization": options.token or options.api_key,
        "User-Agent": _user_agent(),
        "AssemblyAI-Version": "2025-05-12",
    }


def _resolve_options(
    options: Optional[RealTimeTranscriberOptions],
    api_key: Optional[str],
) -> RealTimeTranscriberOptions:
    """Returns the options a streaming client is configured with.

    ``api_key`` takes precedence: given alongside ``options``, it replaces
    the key on a copy of ``options`` and every other field is carried over
    as the caller set it. The caller's ``options`` object is never mutated.

    Args:
        ``options``: an explicit ``RealTimeTranscriberOptions``, or ``None``.
            Returned as-is when no ``api_key`` accompanies it.
        ``api_key``: an API key, or ``None``. On its own it builds options
            with every other field left at its default.

    Raises:
        ValueError: if neither is given.
    """

    if options is not None:
        if api_key is not None:
            # pydantic v2 renamed `copy(update=...)` to `model_copy(update=...)`.
            if hasattr(options, "model_copy"):
                return options.model_copy(update={"api_key": api_key})

            return options.copy(update={"api_key": api_key})

        return options

    if api_key is not None:
        return RealTimeTranscriberOptions(api_key=api_key)

    raise ValueError(
        "Please provide credentials: pass api_key= to the client, or pass "
        "options=RealTimeTranscriberOptions(api_key=...) — or "
        "options=RealTimeTranscriberOptions(token=...) for a temporary token."
    )


class _BaseStreamingClient:
    """Sync/async-agnostic core for streaming clients.

    Subclasses must implement: ``connect``, ``disconnect``, ``stream``,
    ``set_params``, ``force_endpoint``, ``create_temporary_token``, plus
    the I/O loops (``_read_*`` / ``_write_*``). Sync subclasses use plain
    methods; async subclasses use ``async def`` — the return-type
    divergence is why these aren't ``@abstractmethod`` on this base.
    """

    def __init__(self, options: RealTimeTranscriberOptions):
        self._options = options
        self._handlers: Dict[RealTimeEvents, List[Callable]] = {
            event: [] for event in RealTimeEvents.__members__.values()
        }
        # Dedup flags for one-time error dispatch. ``_report_connection_closed``
        # and ``_report_server_error`` perform their flag check + set
        # synchronously (no ``await`` / yield between them) before any
        # dispatch, so even when both I/O tasks/threads race to report the
        # same close only the first caller executes the dispatch body.
        # - Threading: the read thread is the sole dispatcher; the write
        #   thread stages closes via ``_pending_close_error`` for the read
        #   thread to drain.
        # - Asyncio: either task may call the report function; the sync
        #   check-and-set inside the function gives the dedup atomicity.
        self._connection_closed_reported = False
        self._server_error_reported = False
        self._websocket: Optional[Any] = None

    def on(self, event: RealTimeEvents, handler: Callable) -> None:
        """Register a handler for a streaming event.

        ``event`` is a value from ``RealTimeEvents`` (``Begin``, ``Turn``,
        ``Termination``, ``SpeechStarted``, ``Error``, ``Warning``,
        ``LLMGatewayResponse``). ``handler`` is invoked as
        ``handler(client, event)``. For ``AsyncRealTimeTranscriber``, async
        handlers are awaited inline on the read task. Exceptions raised by
        handlers are logged and swallowed — they do not terminate the
        session.
        """
        if event in RealTimeEvents.__members__.values() and callable(handler):
            self._handlers[event].append(handler)

    @staticmethod
    def _parse_event_type(message_type: Optional[Any]) -> Optional[RealTimeEvents]:
        if not isinstance(message_type, str):
            return None
        try:
            return RealTimeEvents[message_type]
        except KeyError:
            return None

    @classmethod
    def _parse_message(cls, data: Dict[str, Any]) -> Optional[EventMessage]:
        if "type" in data:
            event_type = cls._parse_event_type(data.get("type"))

            if event_type == RealTimeEvents.Begin:
                return _parse_model(BeginEvent, data)
            elif event_type == RealTimeEvents.Termination:
                return _parse_model(TerminationEvent, data)
            elif event_type == RealTimeEvents.Turn:
                return _parse_model(TurnEvent, data)
            elif event_type == RealTimeEvents.SpeechStarted:
                return _parse_model(SpeechStartedEvent, data)
            elif event_type == RealTimeEvents.LLMGatewayResponse:
                return _parse_model(LLMGatewayResponseEvent, data)
            elif event_type == RealTimeEvents.SpeakerRevision:
                return _parse_model(SpeakerRevisionEvent, data)
            elif event_type == RealTimeEvents.Heartbeat:
                return _parse_model(HeartbeatEvent, data)
            elif event_type == RealTimeEvents.Error:
                return _parse_model(ErrorEvent, data)
            elif event_type == RealTimeEvents.Warning:
                return _parse_model(WarningEvent, data)
            else:
                return None
        elif "error" in data:
            return _parse_model(ErrorEvent, data)
        return None

    @staticmethod
    def _build_connection_closed_error(
        error: Union[
            RealTimeError,
            ErrorEvent,
            websockets.exceptions.ConnectionClosed,
            OSError,
        ],
    ) -> Optional[RealTimeError]:
        if isinstance(error, RealTimeError):
            return error
        if isinstance(error, ErrorEvent):
            return RealTimeError(message=error.error, code=error.error_code)
        if isinstance(error, websockets.exceptions.ConnectionClosed):
            if error.code == 1000:
                return None
            if error.code is not None and error.code in RealTimeErrorCodes:
                message = RealTimeErrorCodes[error.code]
            else:
                message = error.reason or f"Connection closed (code={error.code})"
            return RealTimeError(message=message, code=error.code)
        return RealTimeError(message=f"Connection failed: {error}")
