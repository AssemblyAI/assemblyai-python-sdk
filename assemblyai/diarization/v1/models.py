"""Models for the Diarization API.

The request options, the result and the error of `/v1/diarize` on
`diarization.assemblyai.com`, alongside the client that uses them.
"""

from typing import List, Optional

from ...types import AssemblyAIError, BaseModel, ConfigDict, Field, pydantic_v2

__all__ = [
    "DiarizationConfig",
    "DiarizationError",
    "DiarizationResponse",
    "DiarizationSegment",
    "DiarizationUtterance",
    "DiarizedWord",
]


class DiarizationError(AssemblyAIError):
    """
    Error raised when a Diarization API request fails.

    Carries a machine-readable `error_code`, the snake_cased problem-details
    `title` from the server (e.g. `unauthorized`, `bad_request`,
    `audio_too_large`), when present, and `retry_after` (seconds) for 429/503
    responses that include a `Retry-After` header. `status_code` is `401` for
    a missing or invalid key and `400` for a request the API rejects
    (an unknown config field, a transcript without `words`).
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


class DiarizationConfig(BaseModel):
    """
    Options for a Diarization API request.

    `sample_rate` and `channels` are required only for raw PCM audio;
    container formats carry them in their own headers.

    Unknown fields are rejected rather than silently dropped: the API
    accepts this exact set and answers `400` to anything else.
    """

    sample_rate: Optional[int] = None
    """Source sample rate in Hz. Setting either this or `channels` marks the
    audio as raw 16-bit PCM, and both are then required."""

    channels: Optional[int] = None
    """Channel count (1 mono, 2 stereo). Setting either this or `sample_rate`
    marks the audio as raw 16-bit PCM, and both are then required."""

    language_code: Optional[str] = None
    """A bare language code such as `en` or `es` (default `en`). Decides
    whether the words of one sentence are voted onto one speaker. Unused
    without a transcript."""

    if pydantic_v2:
        model_config = ConfigDict(extra="forbid")
    else:

        class Config:
            extra = "forbid"


class DiarizedWord(BaseModel):
    """A transcript word with the speaker who said it."""

    text: str
    "The text of the word."

    start: int
    "Word start in milliseconds."

    end: int
    "Word end in milliseconds."

    confidence: Optional[float] = None
    "The word's transcription confidence, when the transcript carried one."

    speaker: str
    "The speaker label: `A`, `B`, and so on."

    speaker_confidence: Optional[float] = None
    "Confidence in the speaker label, 0-1. `None` when unavailable."


class DiarizationUtterance(BaseModel):
    """A run of consecutive words by one speaker."""

    speaker: str
    "The speaker label."

    start: int
    "Start of the first word in milliseconds."

    end: int
    "End of the last word in milliseconds."

    text: str
    "The words' text, joined."

    confidence: Optional[float] = None
    "Mean word confidence."

    speaker_confidence: Optional[float] = None
    "Mean word speaker confidence. `None` unless every word carries one."

    words: List[DiarizedWord] = Field(default_factory=list)
    "The utterance's words."


class DiarizationSegment(BaseModel):
    """One of the model's speaker turns."""

    start: int
    "Segment start in milliseconds."

    end: int
    "Segment end in milliseconds."

    speaker: str
    "The speaker label, the same speaker as in `words`."

    speaker_confidence: Optional[float] = None
    "The segment's calibrated confidence. `None` when unavailable."


class DiarizationResponse(BaseModel):
    """
    The result of a Diarization API request.

    Without a transcript, `words` and `utterances` are empty and `segments`
    and `speakers` are filled. `speakers` lists only the speakers words went
    to when there are words; a speaker no word went to appears in `segments`
    alone.
    """

    words: List[DiarizedWord] = Field(default_factory=list)
    "The transcript's words, each with a speaker."

    utterances: List[DiarizationUtterance] = Field(default_factory=list)
    "Runs of consecutive words by one speaker."

    segments: List[DiarizationSegment] = Field(default_factory=list)
    "The model's speaker turns, sorted by start."

    speakers: List[str] = Field(default_factory=list)
    "The sorted speaker labels."

    audio_duration_ms: int
    "Total audio duration in milliseconds."

    session_id: str
    "Server-generated UUID for this request. Record it to correlate with support."

    request_time_ms: Optional[float] = None
    "End-to-end server-side request time in milliseconds."
