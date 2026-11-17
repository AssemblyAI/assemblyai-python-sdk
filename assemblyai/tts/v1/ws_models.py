"""Models of the Text-to-Speech WebSocket API: the frames a session receives.

The client frames (`Generate`, `Flush`, `Cancel`, `KeepAlive`, `Terminate`) are
plain dicts; the server frames are the events below, dispatched to handlers
registered with `TTSSession.on()`.
"""

import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

try:
    # pydantic v2 import (`model_validator` exists only in v2)
    from pydantic import BaseModel, Field, model_validator  # noqa: F401

    pydantic_v2 = True
except ImportError:
    # pydantic v1 import (fallback for Python < 3.14)
    from pydantic import BaseModel, Field

    pydantic_v2 = False

from .models import TTSError

__all__ = [
    "TTSAudioEvent",
    "TTSBeginEvent",
    "TTSCancelledEvent",
    "TTSEvents",
    "TTSFlushDoneEvent",
    "TTSTerminationEvent",
    "TTSWord",
    "TTSWordBoundariesEvent",
]

# Close codes after which the same session parameters can succeed on a new
# connection, for a close that came without an `Error` frame: a lost
# connection, a transient check failure, a server restart, a server-side
# failure, an expired session and the account's session limit.
_RETRYABLE_CLOSE_CODES = frozenset({1006, 1011, 1012, 3005, 3008, 3009})
# The code of a connection that ended without a close frame.
_ABNORMAL_CLOSURE = 1006

_M = TypeVar("_M", bound=BaseModel)


def _parse(model: Type[_M], data: Dict[str, Any]) -> _M:
    if hasattr(model, "model_validate"):
        return model.model_validate(data)
    return model.parse_obj(data)


class TTSEvents(str, Enum):
    """The server frames of a session, as `TTSSession.on()` names them."""

    Begin = "Begin"
    "Handler receives a `TTSBeginEvent`."
    Audio = "Audio"
    "Handler receives a `TTSAudioEvent`; registering one turns off pulling."
    FlushDone = "FlushDone"
    "Handler receives a `TTSFlushDoneEvent`."
    WordBoundaries = "WordBoundaries"
    "Handler receives a `TTSWordBoundariesEvent`, after the request's `FlushDone`."
    Cancelled = "Cancelled"
    "Handler receives a `TTSCancelledEvent`."
    Termination = "Termination"
    "Handler receives a `TTSTerminationEvent`."
    Error = "Error"
    "Handler receives the `TTSError` that ended the session."

    def __str__(self):
        return self.value


class TTSBeginEvent(BaseModel):
    """The first frame of a session: it is open and configured."""

    id: str
    "The session id. Quote it in support requests."

    expires_at: int
    "Unix time in seconds at which the session's maximum duration ends."

    configuration: Dict[str, Any] = Field(default_factory=dict)
    """The session as created: the effective connect parameters, `model`,
    `voice_id`, `session_ttl_sec`, `limits` and more. An open set; empty
    when the API sends none."""


class TTSAudioEvent(BaseModel):
    """A piece of a request's audio."""

    flush_id: int
    "The request the audio belongs to."

    audio: bytes = Field(..., repr=False)
    "Raw codec bytes (no header), decoded from the frame's base64."


class TTSFlushDoneEvent(BaseModel):
    """A request's audio has all been delivered."""

    flush_id: int
    audio_duration_ms: int
    "The playback duration of the request's audio; `0` for nothing to speak."


class TTSWord(BaseModel):
    word: str
    "One whitespace-delimited token of the request's text, verbatim."
    start: int
    """Milliseconds on the session's whole audio timeline, not the request's:
    subtract the event's `audio_start_ms` for a time within the request."""
    end: int
    confidence: float


class TTSWordBoundariesEvent(BaseModel):
    """
    Word timings of a request: best effort, at most one per request.

    Arrives after the request's `FlushDone`, so usually after its `audio()`
    iteration has ended, and possibly after the next request's first audio;
    match it by `flush_id`. `start` and `end` count from the start of the
    session's audio; `start - audio_start_ms` is the time within the request.
    """

    flush_id: int
    audio_start_ms: int
    "Where the request's audio starts on the session's timeline."
    words: List[TTSWord]


class TTSCancelledEvent(BaseModel):
    """The answer to a `Cancel`."""

    audio_duration_seconds: float
    "Audio streamed for the cancelled request before it stopped."


class TTSTerminationEvent(BaseModel):
    """The session's summary, after `Terminate`."""

    session_duration_seconds: int
    total_input_char_length: int
    "The billed characters."
    audio_duration_seconds: float
    voice_expires_at: Optional[int] = None


def parse_event(data: Dict[str, Any]) -> Any:
    """The event model for a server frame, or `None` for an unknown type."""

    kind = data.get("type")
    if kind == "Audio":
        return TTSAudioEvent(
            flush_id=data["flush_id"], audio=base64.b64decode(data["audio"])
        )
    if kind == "Begin":
        return _parse(
            TTSBeginEvent, {**data, "configuration": data.get("configuration") or {}}
        )
    if kind == "FlushDone":
        return _parse(TTSFlushDoneEvent, data)
    if kind == "WordBoundaries":
        return _parse(TTSWordBoundariesEvent, data)
    if kind == "Cancelled":
        return _parse(TTSCancelledEvent, data)
    if kind == "Termination":
        return _parse(TTSTerminationEvent, data)
    return None


def error_from_frame(data: Dict[str, Any]) -> TTSError:
    """
    The `TTSError` for an `Error` frame. `body` keeps the frame, plus
    `close_code`: the code of the close that follows it (the frame's
    `error_code`), under the same key as for a close without a frame.
    """

    retry_after = data.get("retry_after_seconds")
    body = dict(data)
    body.setdefault("close_code", data.get("error_code"))
    return TTSError(
        data.get("error") or "text-to-speech session failed",
        error_code=data.get("error_type"),
        param=data.get("param"),
        retryable=data.get("retryable"),
        retry_after=int(retry_after) if isinstance(retry_after, (int, float)) else None,
        body=body,
    )


def close_code_and_reason(exc: BaseException) -> Tuple[int, str]:
    """
    The code and reason of the close frame the API sent, from the exception
    that ended the connection; `1006` and no reason when none arrived.
    """

    rcvd = getattr(exc, "rcvd", None)
    if rcvd is None:
        return _ABNORMAL_CLOSURE, ""
    return int(rcvd.code), rcvd.reason or ""


def error_from_close(code: Optional[int], reason: str) -> TTSError:
    """The `TTSError` for a connection closed without a clean `Termination`."""

    if code is None:
        code = _ABNORMAL_CLOSURE
    text = f"Connection closed (code={code})"
    if reason:
        text = f"{text}: {reason}"
    return TTSError(
        text,
        retryable=code in _RETRYABLE_CLOSE_CODES,
        body={"close_code": code, "reason": reason},
    )
