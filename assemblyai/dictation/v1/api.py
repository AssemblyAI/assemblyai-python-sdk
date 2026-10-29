import logging
from typing import Callable, Iterator, Optional, Tuple

import httpx

from ... import types
from ..._multipart import (
    AudioChunks,
    StreamingMultipartEncoder,
    _Aborted,
    iter_chunks,
)
from ...sync.v1.api import _error_from_response as _shared_error_from_response
from .models import DictationError, DictationResponse

logger = logging.getLogger(__name__)

# The endpoint every dictation starts on. The Dictation API serves it under /v1
# only (no unprefixed alias), and also at `/v1/transcribe/stream`, the path it
# shipped under — but `live` is its name. It caps a request at 120 s of audio.
ENDPOINT_TRANSCRIBE_LIVE = "/v1/transcribe/live"
# The buffered endpoint: the whole recording in one body, up to 20 minutes /
# 256 MB. Requested only as the fallback for a live request that ran past the
# live cap; see `transcribe_live(fallback_to_buffered=True)`.
ENDPOINT_TRANSCRIBE = "/v1/transcribe"
ENDPOINT_WARM = "/v1/warm"

# The most audio the fallback holds in memory, matching the buffered
# endpoint's 256 MB body limit. Past it the recording could not be accepted
# anyway, so the copy is dropped and the request fails without being sent.
BUFFERED_MAX_BYTES = 256 * 1024 * 1024

# The live endpoint's answer to audio past its 120 s cap — the one failure the
# buffered fallback retries.
_LIVE_CAP_STATUS = 413
_LIVE_CAP_ERROR_CODE = "audio_too_large"

# Piece size the buffered body is handed to the transport in, so sending the
# copy never makes a second full-size copy of it.
_BUFFERED_SEND_SIZE = 1024 * 1024


def _error_from_response(response: httpx.Response) -> types.AssemblyAIError:
    """
    Builds a `DictationError` from a non-200 response.

    The Dictation API returns the same error envelopes as the sync API — an
    RFC 9457 problem-details body (`{"status", "title", "detail"}`),
    `{"error", "error_code"}`, or a bare `{"detail"}` — so the parsing is
    shared and only the exception class differs.
    """

    return _shared_error_from_response(
        response,
        DictationError,
        "dictation transcription",
    )


def _is_live_cap(error: types.AssemblyAIError) -> bool:
    """Whether a live request failed only because it ran past the 120 s cap."""

    return (
        isinstance(error, DictationError)
        and error.status_code == _LIVE_CAP_STATUS
        and error.error_code == _LIVE_CAP_ERROR_CODE
    )


def _too_large_for_fallback() -> DictationError:
    """The error raised when the recording outgrows `BUFFERED_MAX_BYTES`."""

    return DictationError(
        f"audio exceeds {BUFFERED_MAX_BYTES // (1024 * 1024)} MB, the most the "
        "buffered fallback for a dictation past the live 120 s cap can send",
        status_code=_LIVE_CAP_STATUS,
        error_code=_LIVE_CAP_ERROR_CODE,
    )


class _Recording:
    """
    The copy of the audio kept for the buffered fallback.

    Only audio bytes are copied, never the multipart framing. The copy is
    capped at `BUFFERED_MAX_BYTES`: once it passes that it is dropped, the
    producer is not read further, and a 413 `audio_too_large` is raised. The
    cap holds while the live request is still uploading too — the Dictation
    API reads a live body to its end before answering 413, so most of a long
    recording arrives before the drain — and the live endpoint could never
    accept that much audio anyway (120 s is at most ~23 MB of 16-bit audio).
    """

    def __init__(self) -> None:
        self.audio = bytearray()

    def _keep(self, chunk: bytes) -> None:
        self.audio += chunk
        if len(self.audio) > BUFFERED_MAX_BYTES:
            self.audio = bytearray()
            raise _too_large_for_fallback()


class _RecordedChunks(_Recording):
    """
    The live upload's audio source, keeping a copy of every byte it hands out.

    The transport pulls from it while the live request runs. When that request
    fails on the 120 s cap, `drain()` goes on reading the same source — which
    the transport no longer does — until the producer ends, so the copy holds
    the whole recording for the buffered retry.
    """

    def __init__(self, chunks: AudioChunks) -> None:
        super().__init__()
        self._chunks = iter_chunks(chunks)

    def __iter__(self) -> "_RecordedChunks":
        return self

    def __next__(self) -> bytes:
        chunk = next(self._chunks)
        self._keep(chunk)
        return chunk

    def drain(self) -> None:
        """
        Reads the rest of the source into the copy, until the producer ends.

        Raises:
            DictationError: once the copy passes `BUFFERED_MAX_BYTES`; the copy
                is dropped and the producer is not read further.
            Exception: anything the producer raises, unchanged.
        """
        for chunk in self._chunks:
            self._keep(chunk)


def _buffered_body(
    audio: bytearray,
    *,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
) -> Tuple[str, int, Iterator[bytes]]:
    """
    Frames the complete recording as the buffered endpoint's body.

    Same parts, same order, same audio filename and Content-Type as the live
    upload. The body is handed out in bounded pieces with its length declared
    up front, so the request carries a `Content-Length` rather than chunked
    framing and the recording is never copied whole a second time.

    Returns: `(content_type, content_length, pieces)`.
    """
    encoder = StreamingMultipartEncoder()
    head = encoder.config_part(config) + encoder.audio_header(
        filename, audio_content_type
    )
    tail = encoder.closing()

    def pieces() -> Iterator[bytes]:
        yield head
        for start in range(0, len(audio), _BUFFERED_SEND_SIZE):
            yield bytes(audio[start : start + _BUFFERED_SEND_SIZE])
        yield tail

    return encoder.content_type, len(head) + len(audio) + len(tail), pieces()


def _parse(response: httpx.Response) -> DictationResponse:
    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DictationResponse.parse_obj(response.json())


def transcribe_buffered(
    client: httpx.Client,
    *,
    base_url: str,
    audio: bytearray,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
) -> DictationResponse:
    """
    Posts a complete recording to the buffered endpoint, `/v1/transcribe`.

    The fallback for a live request that ran past the 120 s cap: the buffered
    endpoint takes up to 20 minutes / 256 MB, the same config fields, and
    answers with the same response shape.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Dictation API base URL.
        audio: the whole recording (WAV container or S16LE PCM).
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        config: the JSON `config` part, or None for an empty one.
        timeout: per-operation timeout in seconds.

    Returns: the parsed dictation response.

    Raises: `DictationError` on any non-200 response.
    """
    content_type, length, pieces = _buffered_body(
        audio,
        filename=filename,
        audio_content_type=audio_content_type,
        config=config,
    )
    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        content=pieces,
        headers={"Content-Type": content_type, "Content-Length": str(length)},
        timeout=timeout,
    )

    return _parse(response)


def transcribe_live(
    client: httpx.Client,
    *,
    base_url: str,
    chunks: AudioChunks,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
    fallback_to_buffered: bool = False,
    cancelled: Optional[Callable[[], bool]] = None,
) -> DictationResponse:
    """
    Posts a dictation request whose audio is uploaded as it arrives.

    Sends the body with chunked transfer encoding — httpx frames an unsized
    iterator that way — so the request can start before the audio exists. The
    server transcribes each speech segment as it lands, leaving only the final
    segment's inference (and the LLM pass, when one was asked for) to wait on
    once the caller stops speaking. Audio that is already complete travels the
    same way, as a single chunk: there is one request shape in this client.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Dictation API base URL, e.g. `https://dictation.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        config: the JSON `config` part. None sends an empty object: the live
            endpoint requires the part ahead of the audio.
        timeout: per-operation timeout in seconds, as for every httpx request:
            it bounds connecting, each socket write and each read while waiting
            for the response, not the request end to end. Time blocked in the
            caller's producer is not counted, so it need not cover the
            recording. The buffered fallback request gets the same timeout;
            draining the producer before it is not bounded by any timeout.
        fallback_to_buffered: keep a copy of the audio and, if the live request
            fails with 413 `audio_too_large` (the 120 s cap), read the producer
            to its end and post the whole recording to `/v1/transcribe`.
        cancelled: polled once the producer has ended, before the fallback
            request; when it returns True the request is dropped instead.

    Returns: the parsed dictation response, from the live endpoint or, after
        a fallback, from the buffered one.

    Raises: `DictationError` on any non-200 response — including one the
        server sends while the upload is still in flight, which it may do for
        auth, rate-limit, size and capacity failures. After a fallback it is
        the buffered request's own error, or a 413 `audio_too_large` raised
        here once the recording passes `BUFFERED_MAX_BYTES`. `TypeError` if
        the producer yields something other than bytes. Any other exception the
        producer raises propagates unchanged; the connection is dropped and no
        transcript is returned.
    """
    encoder = StreamingMultipartEncoder()
    recorded = _RecordedChunks(chunks) if fallback_to_buffered else None
    source: Iterator[bytes] = recorded if recorded is not None else iter_chunks(chunks)

    def body() -> Iterator[bytes]:
        # config first: the server needs sample_rate and channels before it can
        # decode a single audio byte, and rejects audio that arrives first.
        yield encoder.config_part(config) + encoder.audio_header(
            filename, audio_content_type
        )

        for chunk in source:
            if chunk:
                yield chunk

        yield encoder.closing()

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE_LIVE,
        content=body(),
        headers={"Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code == httpx.codes.OK:
        return DictationResponse.parse_obj(response.json())

    error = _error_from_response(response)
    if recorded is None or not _is_live_cap(error):
        raise error

    # Past the live cap. The 413 may come back while the producer is still
    # recording — the transport stops reading the body once it has the
    # response — so keep reading it until the speaker stops. That wait is
    # recording time, not network time, and no timeout bounds it.
    recorded.drain()
    if cancelled is not None and cancelled():
        raise _Aborted()

    audio = recorded.audio
    logger.info(
        "Dictation passed the live endpoint's 120 s cap; posting the whole "
        "recording (%d bytes) to %s",
        len(audio),
        ENDPOINT_TRANSCRIBE,
    )

    return transcribe_buffered(
        client,
        base_url=base_url,
        audio=audio,
        filename=filename,
        audio_content_type=audio_content_type,
        config=config,
        timeout=timeout,
    )
