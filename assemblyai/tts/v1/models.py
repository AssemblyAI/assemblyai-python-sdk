"""Models for the Text-to-Speech API.

The request options, the audio result and the error of `/v1/speech` on
`tts.us.assemblyai.com` (or `tts.eu.assemblyai.com`), alongside the client
that uses them.
"""

from enum import Enum
from typing import List, Optional, Union

from ...types import AssemblyAIError, pydantic_v2

# pydantic 1.10 also exports `ConfigDict`, so the version is taken from
# `types.py`, which probes a v2-only name, instead of an import attempt here.
if pydantic_v2:
    from pydantic import BaseModel, ConfigDict, Field
else:
    from pydantic import BaseModel, Field

__all__ = [
    "TTSConfig",
    "TTSContainer",
    "TTSEncoding",
    "TTSError",
    "TTSResponse",
    "TTSResponseInfo",
]


class TTSError(AssemblyAIError):
    """
    Error raised when a Text-to-Speech request fails.

    `error_code` is the API's stable `code` (`invalid_parameter`,
    `voice_not_found`, `rate_limited`, `at_capacity`, ...): branch on it, not
    on the message, which may be reworded. `retryable` says whether the same
    request can succeed later and `retry_after` (seconds) when to send it;
    `param` names the field at fault and `errors` lists every failing field
    (`{"message", "param"}`) of an `invalid_parameter` refusal. `request_id`
    is the value to quote to support.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        param: Optional[str] = None,
        request_id: Optional[str] = None,
        retryable: Optional[bool] = None,
        retry_after: Optional[int] = None,
        errors: Optional[List[dict]] = None,
    ):
        super().__init__(message, status_code)
        self.error_code = error_code
        self.param = param
        self.request_id = request_id
        self.retryable = retryable
        self.retry_after = retry_after
        self.errors = errors


class TTSEncoding(str, Enum):
    """Codec of the audio the API returns."""

    pcm_s16le = "pcm_s16le"
    "The default: mono signed 16-bit little-endian PCM."

    pcm_mulaw = "pcm_mulaw"
    "8-bit G.711 mu-law, for telephony."

    pcm_alaw = "pcm_alaw"
    "8-bit G.711 A-law, for telephony."

    mp3 = "mp3"
    "Constant bit rate MP3; see `TTSConfig.bit_rate`."

    def __str__(self):
        return self.value


class TTSContainer(str, Enum):
    """Container of a PCM or G.711 response."""

    raw = "raw"
    "The default: the codec bytes with no header."

    wav = "wav"
    "A WAV file. On a streamed response its size fields are unknown (all ones)."

    def __str__(self):
        return self.value


class TTSConfig(BaseModel):
    """
    Options for a Text-to-Speech request.

    Every field is optional: the API speaks English with its default voice at
    24 kHz as raw 16-bit PCM when nothing is set. Unknown fields are rejected
    rather than silently dropped, so a typo surfaces as a validation error.
    """

    voice: Optional[str] = None
    """The name of an AssemblyAI voice: `jane` (the default), `alba`, `anna`,
    `charles`, `eve`, `george`, `jean`, `mary`, `michael`, `paul` or `vera`,
    each in every language. Case-insensitive. Not together with `voice_id`."""

    voice_id: Optional[str] = None
    """A voice by its id: a voice cloned by the account (its UUID or slug) or a
    voice of the AssemblyAI voice library by its global slug, such as
    `aai-jane`. Sets the request's language to the voice's own unless
    `language` is given. Not together with `voice`."""

    model_id: Optional[str] = None
    """The model to run the request on, such as `assemblyai-tts-1`. Defaults to
    the region's default model of the request's language."""

    language: Optional[str] = None
    """The language, by its English name: `english` (the default), `spanish`,
    `german`, `french`, `italian` or `portuguese`. Case-insensitive."""

    sample_rate: Optional[int] = None
    "Output sample rate in Hz: 8000, 16000, 22050, 24000 (the default), 44100 or 48000."

    encoding: Optional[Union[TTSEncoding, str]] = None
    """Output codec: a `TTSEncoding`, or its name as a string. The API also
    takes the aliases `mulaw` and `alaw`. Defaults to `pcm_s16le`."""

    container: Optional[TTSContainer] = None
    "`raw` (the default) or `wav`, with a PCM or G.711 encoding only."

    bit_rate: Optional[int] = None
    """MP3 bit rate in bits per second: 32000, 48000, 64000 (the default),
    96000 or 128000; at most 64000 at 8 kHz. With `encoding=mp3` only."""

    extend_biometric_consent: Optional[bool] = None
    """`True` renews the biometric consent of the cloned voice in `voice_id`,
    which only the voice's owner may do. A cloned voice expires 60 days after
    its consent was last given."""

    if pydantic_v2:
        model_config = ConfigDict(extra="forbid")

    else:

        class Config:
            extra = "forbid"


class TTSResponseInfo(BaseModel):
    """
    What the response headers say about the audio of a request.

    Every field is `None` when the API did not send its header.
    """

    request_id: Optional[str] = None
    "The request's id, from `X-Request-Id`. Quote it in support requests."

    model: Optional[str] = None
    "The model that spoke, such as `assemblyai-tts-1`."

    voice: Optional[str] = None
    "The voice that spoke, such as `jane`; `custom` for a voice named by `voice_id`."

    language: Optional[str] = None
    "The language used, lowercase."

    encoding: Optional[str] = None
    "The codec used, in canonical form (`pcm_mulaw`, never `mulaw`)."

    sample_rate: Optional[int] = None
    "The sample rate used, in Hz."

    container: Optional[str] = None
    "`raw` or `wav`; `None` for MP3."

    bit_rate: Optional[int] = None
    "The MP3 bit rate in bits per second; `None` otherwise."

    content_type: Optional[str] = None
    "The response's `Content-Type`, such as `audio/pcm;rate=24000;channels=1`."

    voice_expires_at: Optional[int] = None
    """For a cloned voice of the account that expires: its expiry as a Unix
    time in seconds, after the request's consent renewal if it asked for one.
    `None` for a preset or library voice."""

    billed_characters: Optional[int] = None
    "The characters the request bills."

    audio_duration_ms: Optional[int] = None
    """The duration of the audio in milliseconds. Sent on a buffered response
    (`synthesize()`) only: a streamed response does not know it up front."""


class TTSResponse(TTSResponseInfo):
    """The result of `TextToSpeech.synthesize()`: the whole audio, with its headers."""

    audio: bytes = Field(..., repr=False)
    """The audio in the requested encoding and container, complete. Its
    duration is `audio_duration_ms`. Left out of `repr()`, so a logged
    response does not carry the clip."""
