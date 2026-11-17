"""Models for the Text-to-Speech API.

The request options, the audio result and the error of `/v1/speech` on
`tts.us.assemblyai.com` (or `tts.eu.assemblyai.com`), alongside the client
that uses them.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    # pydantic v2 import (`model_validator` exists only in v2; pydantic 1.10
    # also exports `ConfigDict`, so it is not the probe)
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    pydantic_v2 = True
except ImportError:
    # pydantic v1 import (fallback for Python < 3.14)
    from pydantic import BaseModel, Field, root_validator

    pydantic_v2 = False

from ...types import AssemblyAIError

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
    `voice_not_found`, `biometric_terms_not_accepted`, `rate_limited`,
    `at_capacity`, `auth_unavailable`, ...): branch on it, not on the message,
    which may be reworded. `retryable` says whether the same request can
    succeed later and `retry_after` (seconds) when to send it; `param` names
    the field at fault and `errors` lists every failing field
    (`{"message", "param"}`) of an `invalid_parameter` refusal. `request_id`
    is the value to quote to support.

    `body` is the parsed error envelope as the API sent it, for fields some
    codes add (`rate_limited` adds `limit` and `limit_window`; a text or
    body over its limit adds `limit`).

    `biometric_terms_not_accepted` (`403`, not retryable) refuses one of the
    account's own cloned voices, an inline voice clip or
    `extend_biometric_consent`, until the account accepts the biometric data
    terms in the AssemblyAI dashboard. `auth_unavailable` (`503`, retryable)
    means the credential, or those terms, could not be checked just now.

    On a WebSocket session (`TTSSession`) there is no HTTP status:
    `status_code` is `None`, `request_id` is the session id, `retryable` is
    the API's (from its `Error` frame, else from the close code), and
    `body["close_code"]` is the WebSocket close code. There
    `biometric_terms_not_accepted` arrives as close `3006` and
    `auth_unavailable` as close `1011`.

    A streamed response cut after its first audio raises a `TTSError` with
    `status_code` `200`, no `error_code`, `retryable` `True` and the
    request's `request_id`; the httpx error is its `__cause__`. Retry the
    whole request.
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
        errors: Optional[List[Dict[str, Optional[str]]]] = None,
        body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, status_code)
        self.error_code = error_code
        self.param = param
        self.request_id = request_id
        self.retryable = retryable
        self.retry_after = retry_after
        self.errors = errors
        self.body = body


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


def _refuse_voice_field(data: Any) -> Any:
    """A `voice` key is refused with a message that points to `voice_id`."""

    if isinstance(data, dict) and "voice" in data:
        raise ValueError(
            "TTSConfig has no field `voice`: name the voice with `voice_id`, "
            'such as voice_id="aai-jane"'
        )
    return data


class TTSConfig(BaseModel):
    """
    Options for a Text-to-Speech request.

    Every field is optional: the API speaks English with its default voice
    (`aai-jane`) at 24 kHz as raw 16-bit PCM when nothing is set. Every voice
    is named by `voice_id`. Unknown fields are rejected rather than silently
    dropped, so a typo surfaces as a validation error. `speed` (only 1 is
    supported) and `word_boundaries` (served by `/v1/speech/events`) are
    left out on purpose.
    """

    voice_id: Optional[str] = None
    """The voice: a voice of the AssemblyAI voice library by its slug, such as
    `aai-jane` (the default), `aai-alba` or `aai-michael`; a voice's ID (a
    UUID); or the slug of one of the account's own cloned voices. The
    dashboard shows the value that names each voice. A voice belongs to the
    data zone it was created in: on the other zone's host it is
    `voice_not_found`."""

    model_id: Optional[str] = None
    """The model to run the request on, such as `assemblyai-tts-1`. Defaults to
    the region's default model of the request's language. A cloned voice must
    be paired with it."""

    language: Optional[str] = None
    """The language, by its English name: `english`, `spanish`, `german`,
    `french`, `italian` or `portuguese`. Case-insensitive. Defaults to the
    voice's own language. The eleven built-in voices (`aai-jane`, `aai-alba`,
    `aai-anna`, `aai-charles`, `aai-eve`, `aai-george`, `aai-jean`,
    `aai-mary`, `aai-michael`, `aai-paul`, `aai-vera`) speak all six; another
    voice speaks a language other than its own only where its model can."""

    sample_rate: Optional[int] = None
    "Output sample rate in Hz: 8000, 16000, 22050, 24000 (the default), 44100 or 48000."

    encoding: Optional[Union[TTSEncoding, str]] = None
    """Output codec: a `TTSEncoding`, or its name as a string. The API also
    takes the aliases `mulaw` and `alaw`. Defaults to `pcm_s16le`."""

    container: Optional[Union[TTSContainer, str]] = None
    """`raw` (the default) or `wav`, with a PCM or G.711 encoding only: a
    `TTSContainer`, or its name as a string."""

    bit_rate: Optional[int] = None
    """MP3 bit rate in bits per second: 32000, 48000, 64000 (the default),
    96000 or 128000; at most 64000 at 8 kHz. With `encoding=mp3` only."""

    extend_biometric_consent: Optional[bool] = None
    """`True` renews the biometric consent of the cloned voice in `voice_id`,
    which only the voice's owner may do; `voice_expires_at` on the response
    shows the new expiry. A cloned voice expires 60 days after its consent
    was last given. Refused `not_owner` on a voice the account does not own
    and `403 biometric_terms_not_accepted` until the account has accepted the
    biometric data terms in the dashboard."""

    if pydantic_v2:
        # `model_id` is the API's field name: `protected_namespaces=()` keeps
        # pydantic 2.0 to 2.9 from warning about its `model_` prefix on import.
        model_config = ConfigDict(extra="forbid", protected_namespaces=())

        @model_validator(mode="before")
        @classmethod
        def _refuse_voice(cls, data: Any) -> Any:
            return _refuse_voice_field(data)

    else:

        class Config:
            extra = "forbid"

        @root_validator(pre=True, allow_reuse=True)
        def _refuse_voice(cls, values: Any) -> Any:
            return _refuse_voice_field(values)


class TTSResponseInfo(BaseModel):
    """
    What the response headers say about the audio of a request.

    Every field is `None` when the API did not send its header.
    """

    request_id: Optional[str] = None
    "The request's id, from `X-Request-Id`. Quote it in support requests."

    region: Optional[str] = None
    "The region that served the request, such as `us-east-1`."

    model: Optional[str] = None
    "The model that spoke, such as `assemblyai-tts-1`."

    voice_id: Optional[str] = None
    """The voice that spoke: the request's `voice_id`, or the default voice's
    slug (`aai-jane`) when the request named none."""

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
    `None` for a library voice. On a stream, the value as the stream started."""

    billed_characters: Optional[int] = None
    """The characters the request bills. On a stream, the value as the stream
    started: a body cut by a server failure bills nothing, one cut by the
    API's 600-second limit (`session_expired`) bills only the text whose
    audio had been generated, and one cut by a client reading too slowly
    (`audio_backlog`) bills every character."""

    audio_duration_ms: Optional[int] = None
    """The duration of the audio in milliseconds, on a complete response:
    `synthesize()`, and a stream whose text had nothing to speak (`0`). A
    stream with audio does not know it up front: `None`."""


class TTSResponse(TTSResponseInfo):
    """The result of `TextToSpeech.synthesize()`: the whole audio, with its headers."""

    audio: bytes = Field(..., repr=False)
    """The audio in the requested encoding and container, complete. Its
    duration is `audio_duration_ms`. Empty (or a header-only WAV) for a text
    with nothing to speak, such as one made only of emoji, which bills `0`.
    Left out of `repr()`, so a logged response does not carry the clip."""
