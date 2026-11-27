"""Models for the Dictation API.

The request and response shapes of `dictation.assemblyai.com`, plus the error
the transcribers raise, live here alongside the client that uses them.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    # pydantic v2 import (`model_validator` exists only in v2; pydantic 1.10
    # also exports `ConfigDict`, so it is not the probe)
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        field_validator,
        model_validator,
    )

    pydantic_v2 = True
except ImportError:
    # pydantic v1 import (fallback for Python < 3.14)
    from pydantic import BaseModel, Field, root_validator, validator

    pydantic_v2 = False

from ...types import AssemblyAIError

__all__ = [
    "DictationConfig",
    "DictationError",
    "DictationLLMRequest",
    "DictationResponse",
    "DictationRewriteMode",
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


_DICTATION_MAX_KEYTERMS_PROMPT_LEN = 8000
_DICTATION_MAX_KEYTERMS_COUNT = 100
_DICTATION_MAX_LLM_INSTRUCTION_LEN = 2048
_DICTATION_MAX_STT_PROMPT_LEN = 6000
_DICTATION_MAX_LLM_OUTPUT_TOKENS = 8192
_DICTATION_RESERVED_PROMPT_REPLACEMENTS = ("text", "transcript")


class DictationRewriteMode(str, Enum):
    """Model tier that runs the transcript rewrite."""

    fast = "fast"
    "The default: the lowest-latency rewrite."

    smart = "smart"
    """Follows detailed formatting instructions more reliably, at higher
    latency."""

    def __str__(self):
        return self.value


def _check_llm_request(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Raises on the `llm` combinations the Dictation API rejects with a 400.

    A saved prompt runs as saved, so only `prompt_replacements` may come with
    `aai_prompt`, and a replacement cannot take a name Dictation or LLM
    Gateway fills itself.
    """

    replacements = values.get("prompt_replacements")
    if values.get("aai_prompt") is None:
        if replacements is not None:
            raise ValueError("prompt_replacements requires aai_prompt")
        return values

    extra = sorted(
        name
        for name, value in values.items()
        if value is not None and name not in ("aai_prompt", "prompt_replacements")
    )
    if extra:
        raise ValueError(
            f"a saved prompt runs as saved; drop {', '.join(extra)} or aai_prompt"
        )

    for key in _DICTATION_RESERVED_PROMPT_REPLACEMENTS:
        if replacements and key in replacements:
            raise ValueError(
                f"prompt_replacements cannot set {key}: Dictation fills `text` "
                "with the transcript and LLM Gateway reserves `transcript`"
            )
    return values


class DictationLLMRequest(BaseModel):
    """
    An LLM Gateway request Dictation runs over the transcript, in place of
    `llm_instruction`.

    Either write the prompt yourself in `messages` or `prompt`, with
    `{{turn}}` where the transcript goes, or name a saved prompt in
    `aai_prompt` (it gets the transcript as `{{ text }}`). Every field is
    optional, and unset fields are not sent. The rewrite comes back in
    `DictationResponse.llm_response`.

    Unknown fields are rejected: Dictation accepts this exact set and returns
    a 400 for any other field, such as `fallbacks`, `tools`, `stream` or
    `transcript_id`.
    """

    model: Optional[str] = None
    """The LLM Gateway model. It must be one the Dictation rewrite runs on;
    without it, `rewrite_mode` picks one. Not allowed with `aai_prompt`."""

    messages: Optional[List[Dict[str, Any]]] = None
    """Chat messages, as in an LLM Gateway chat completions request. Put
    `{{turn}}` where the transcript goes."""

    prompt: Optional[str] = None
    "A single prompt, in place of `messages`. Put `{{turn}}` where the transcript goes."

    temperature: Optional[float] = None
    "Sampling temperature. Dictation uses 0 unless this is set."

    top_p: Optional[float] = None
    "Nucleus sampling cutoff."

    top_k: Optional[int] = None
    "Top-k sampling cutoff."

    max_tokens: Optional[int] = Field(default=None, le=_DICTATION_MAX_LLM_OUTPUT_TOKENS)
    """Output token limit, at most 8192. Unset, Dictation sizes it from the
    transcript, so a prompt that expands the text may need a larger one. A
    rewrite that runs out comes back with `llm_error` "truncated"."""

    max_completion_tokens: Optional[int] = Field(
        default=None, le=_DICTATION_MAX_LLM_OUTPUT_TOKENS
    )
    "Output token limit, at most 8192. The same limit under its other name."

    response_format: Optional[Dict[str, Any]] = None
    "The LLM Gateway `response_format`, e.g. a JSON schema for the output."

    reasoning_effort: Optional[str] = None
    "How much the model reasons before it answers, for models that support it."

    aai_prompt: Optional[str] = None
    """The name of a prompt saved in your account. The saved prompt's model
    and settings apply, so only `prompt_replacements` may accompany it, and
    `rewrite_mode` may not be set. An unknown name comes back as `llm_error`
    "prompt_not_found"."""

    prompt_replacements: Optional[Dict[str, Any]] = None
    """Values for the `{{ tag }}` placeholders of the saved prompt. Requires
    `aai_prompt`. `text` (the transcript) and `transcript` are reserved."""

    zero_data_retention: Optional[bool] = None
    """Accepted for parity with LLM Gateway. Dictation sets zero data
    retention on every rewrite request regardless."""

    if pydantic_v2:
        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="after")
        def _check_rules(self) -> "DictationLLMRequest":
            _check_llm_request(dict(self))
            return self

    else:

        class Config:
            extra = "forbid"

        @root_validator(skip_on_failure=True, allow_reuse=True)
        def _check_rules(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            return _check_llm_request(values)


def _check_rewrite_source(values: Dict[str, Any]) -> Dict[str, Any]:
    """Raises when the config asks for the rewrite two ways at once."""

    llm = values.get("llm")
    if llm is None:
        return values
    if values.get("llm_instruction") is not None:
        raise ValueError("set either llm_instruction or llm, not both")
    if llm.aai_prompt is not None and values.get("rewrite_mode") is not None:
        raise ValueError(
            "rewrite_mode does not apply to a saved prompt (aai_prompt): "
            "the saved prompt picks its own model"
        )
    return values


class DictationConfig(BaseModel):
    """
    Options for a Dictation API request.

    `language_codes`, `stt_prompt` and `keyterms_prompt` shape the
    transcript; `llm_instruction` asks the server to run a follow-up LLM pass over it
    (the rewrite lands in `DictationResponse.llm_response`), and `llm` sends
    an LLM Gateway request of your own for that pass instead. `sample_rate`
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

    stt_prompt: Optional[str] = Field(
        default=None, max_length=_DICTATION_MAX_STT_PROMPT_LEN
    )
    """Context for the transcription: a description of what the audio is
    about, e.g. "A doctor dictating a patient visit note." It describes the
    situation rather than instructing the model, and steers the decoder as it
    writes the transcript — where `llm_instruction` reshapes the transcript
    afterwards. Max 6000 characters."""

    keyterms_prompt: Optional[List[str]] = None
    "Keyterms biasing the decoder. Whitespace is stripped and empty terms dropped. Max 100 terms / 8000 characters total."

    llm_instruction: Optional[str] = Field(
        default=None, max_length=_DICTATION_MAX_LLM_INSTRUCTION_LEN
    )
    """Instruction for a follow-up LLM pass over the transcript, e.g.
    "Format this as a SOAP note." The rewritten text comes back as
    `llm_response`; the raw transcript stays in `text`. Max 2048 characters."""

    llm: Optional[DictationLLMRequest] = None
    """An LLM Gateway request of your own for the rewrite, in place of
    `llm_instruction`: your own `messages` or `prompt` with `{{turn}}` where
    the transcript goes, or a saved prompt by `aai_prompt`. A plain dict is
    accepted. Set either this or `llm_instruction`, not both. The rewrite
    comes back as `llm_response`."""

    rewrite_mode: Optional[DictationRewriteMode] = None
    """Model tier that runs the rewrite: `fast` or `smart`. `smart` follows
    detailed formatting instructions more reliably at higher latency. Defaults
    to None, which leaves the server default (`fast`)."""

    if pydantic_v2:
        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="after")
        def _check_rewrite(self) -> "DictationConfig":
            _check_rewrite_source(dict(self))
            return self

        @field_validator("keyterms_prompt")
        @classmethod
        def _normalize_keyterms_prompt(cls, v):
            if not v:
                return None
            terms = [t.strip() for t in v if t and t.strip()]
            if len(terms) > _DICTATION_MAX_KEYTERMS_COUNT:
                raise ValueError(
                    f"keyterms_prompt exceeds {_DICTATION_MAX_KEYTERMS_COUNT} terms (got {len(terms)})"
                )
            total = sum(len(t) for t in terms)
            if total > _DICTATION_MAX_KEYTERMS_PROMPT_LEN:
                raise ValueError(
                    f"keyterms_prompt exceeds {_DICTATION_MAX_KEYTERMS_PROMPT_LEN} characters (got {total})"
                )
            return terms or None

    else:

        class Config:
            extra = "forbid"

        @root_validator(skip_on_failure=True, allow_reuse=True)
        def _check_rewrite(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            return _check_rewrite_source(values)

        @validator("keyterms_prompt")
        def _normalize_keyterms_prompt(cls, v):
            if not v:
                return None
            terms = [t.strip() for t in v if t and t.strip()]
            if len(terms) > _DICTATION_MAX_KEYTERMS_COUNT:
                raise ValueError(
                    f"keyterms_prompt exceeds {_DICTATION_MAX_KEYTERMS_COUNT} terms (got {len(terms)})"
                )
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
    """The transcript rewritten by the LLM pass `llm_instruction` or `llm`
    asked for. `None` when neither was sent or the pass failed."""

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

    auth_time_ms: Optional[float] = None
    """Authentication and key-validation time in milliseconds, as a portion
    of request_time_ms. ``None`` when the server predates the field."""

    @property
    def final_text(self) -> str:
        """
        The text to show the user: the LLM rewrite when there is one.

        Falls back to the raw transcript when neither `llm_instruction` nor
        `llm` was sent or the LLM pass failed, so reading it is safe either way.
        """

        return self.llm_response if self.llm_response is not None else self.text
