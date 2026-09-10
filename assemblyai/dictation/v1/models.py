"""Models for the Dictation API.

The request and response shapes of `dictation.assemblyai.com`, plus the error
the transcribers raise, live here alongside the client that uses them.
"""

from typing import List, Optional

try:
    # pydantic v2 import
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    pydantic_v2 = True
except ImportError:
    # pydantic v1 import (fallback for Python < 3.14)
    from pydantic import BaseModel, Field, validator

    pydantic_v2 = False

from ...types import AssemblyAIError

__all__ = [
    "DictationConfig",
    "DictationError",
    "DictationResponse",
    "DictationWord",
]


class DictationError(AssemblyAIError):
    """
    Error raised when a Dictation API request fails.

    Carries a machine-readable `error_code` — the snake_cased problem-details
    `title` from the server (e.g. `bad_audio`, `audio_too_large`,
    `capacity_exceeded`, `inference_timeout`) — when present, and
    `retry_after` (seconds) for 429/503 responses that include a
    `Retry-After` header.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message, status_code)
        self.error_code = error_code
        self.retry_after = retry_after


_DICTATION_MAX_KEYTERMS_PROMPT_LEN = 2048
_DICTATION_MAX_LLM_INSTRUCTION_LEN = 2048


class DictationConfig(BaseModel):
    """
    Options for a Dictation API request.

    `language_codes` and `keyterms_prompt` shape the transcript;
    `llm_instruction` asks the server to run a follow-up LLM pass over it
    (the rewrite lands in `DictationResponse.llm_response`). `sample_rate`
    and `channels` are required only for raw PCM audio — container formats
    carry them in their own headers.

    Unknown fields are rejected rather than silently dropped: the Dictation
    API accepts this exact set, so a typo or a sync-only option surfaces as a
    validation error instead of a config that quietly does nothing.
    """

    sample_rate: Optional[int] = None
    """Source sample rate in Hz. Setting either this or `channels` marks the
    audio as raw 16-bit PCM, and both are then required. Leave both unset for
    WAV and the other container formats, which carry the rate in their own
    headers."""

    channels: Optional[int] = None
    """Channel count (1 mono, 2 stereo). Setting either this or `sample_rate`
    marks the audio as raw 16-bit PCM, and both are then required. Leave both
    unset for WAV and the other container formats, which carry the channel
    count in their own headers."""

    language_codes: Optional[List[str]] = None
    """ISO 639-1 codes for the language(s) of the audio — a single-element
    list (e.g. `["es"]`) for monolingual audio, or several codes (e.g.
    `["en", "es"]`) for multilingual audio. Defaults to None, which leaves
    the language to the server's default."""

    keyterms_prompt: Optional[List[str]] = None
    "Keyterms biasing the decoder. Whitespace is stripped and empty terms dropped. Max 2048 characters total."

    llm_instruction: Optional[str] = Field(
        default=None, max_length=_DICTATION_MAX_LLM_INSTRUCTION_LEN
    )
    """Instruction for a follow-up LLM pass over the transcript, e.g.
    "Format this as a SOAP note." The rewritten text comes back as
    `llm_response`; the raw transcript stays in `text`. Max 2048 characters."""

    if pydantic_v2:
        model_config = ConfigDict(extra="forbid")

        @field_validator("keyterms_prompt")
        @classmethod
        def _normalize_keyterms_prompt(cls, v):
            if not v:
                return None
            terms = [t.strip() for t in v if t and t.strip()]
            total = sum(len(t) for t in terms)
            if total > _DICTATION_MAX_KEYTERMS_PROMPT_LEN:
                raise ValueError(
                    f"keyterms_prompt exceeds {_DICTATION_MAX_KEYTERMS_PROMPT_LEN} characters (got {total})"
                )
            return terms or None

    else:

        class Config:
            extra = "forbid"

        @validator("keyterms_prompt")
        def _normalize_keyterms_prompt(cls, v):
            if not v:
                return None
            terms = [t.strip() for t in v if t and t.strip()]
            total = sum(len(t) for t in terms)
            if total > _DICTATION_MAX_KEYTERMS_PROMPT_LEN:
                raise ValueError(
                    f"keyterms_prompt exceeds {_DICTATION_MAX_KEYTERMS_PROMPT_LEN} characters (got {total})"
                )
            return terms or None


class DictationWord(BaseModel):
    """A single word in a dictation transcript."""

    text: str
    "The text of the word."

    confidence: float
    "Word confidence in the range 0-1."


class DictationResponse(BaseModel):
    """The result of a Dictation API request."""

    text: str
    "The raw transcript text, before any LLM pass."

    words: List[DictationWord] = Field(default_factory=list)
    "Per-word text and confidence."

    confidence: float
    "Overall transcript confidence in the range 0-1."

    llm_response: Optional[str] = None
    """The transcript rewritten by the LLM pass `llm_instruction` asked for.
    `None` when no instruction was sent or the pass failed."""

    llm_error: Optional[str] = None
    "Why the LLM pass failed, when it did. `None` otherwise."

    audio_duration_ms: int
    "Total audio duration in milliseconds."

    session_id: str
    "Server-generated UUID for this request. Record it to correlate with support."

    request_time_ms: Optional[float] = None
    """End-to-end server-side request time in milliseconds: auth, multipart
    parse, decode, inference, the LLM pass, and serialization."""

    sync_time_ms: Optional[float] = None
    "Time in milliseconds spent transcribing, excluding the LLM pass."

    @property
    def final_text(self) -> str:
        """
        The text to show the user: the LLM rewrite when there is one.

        Falls back to the raw transcript when no `llm_instruction` was sent
        or the LLM pass failed, so reading it is safe either way.
        """

        return self.llm_response if self.llm_response is not None else self.text
