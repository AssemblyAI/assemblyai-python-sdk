from typing import Iterator, Optional

import httpx

from ... import types
from ..._multipart import AudioChunks, StreamingMultipartEncoder, iter_chunks
from ...sync.v1.api import _error_from_response as _shared_error_from_response
from .models import DictationError, DictationResponse

# The one endpoint this client posts audio to. The Dictation API serves it
# under /v1 only (no unprefixed alias), and also at `/v1/transcribe/stream`,
# the path it shipped under — but `live` is its name.
ENDPOINT_TRANSCRIBE_LIVE = "/v1/transcribe/live"
ENDPOINT_WARM = "/v1/warm"


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


def transcribe_live(
    client: httpx.Client,
    *,
    base_url: str,
    chunks: AudioChunks,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
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
            recording.

    Returns: the parsed dictation response.

    Raises: `DictationError` on any non-200 response — including one the
        server sends while the upload is still in flight, which it may do for
        auth, rate-limit, size and capacity failures. `TypeError` if the
        producer yields something other than bytes. Any other exception the
        producer raises propagates unchanged; the connection is dropped and no
        transcript is returned.
    """
    encoder = StreamingMultipartEncoder()

    def body() -> Iterator[bytes]:
        # config first: the server needs sample_rate and channels before it can
        # decode a single audio byte, and rejects audio that arrives first.
        yield encoder.config_part(config) + encoder.audio_header(
            filename, audio_content_type
        )

        for chunk in iter_chunks(chunks):
            if chunk:
                yield chunk

        yield encoder.closing()

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE_LIVE,
        content=body(),
        headers={"Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DictationResponse.parse_obj(response.json())
