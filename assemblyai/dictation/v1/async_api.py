"""The asyncio counterpart of `api.py`.

Calls the same endpoint as its sync twin and raises the same `DictationError`
through `api._error_from_response`.
"""

from typing import AsyncIterator, Optional

import httpx

from ..._multipart import AsyncAudioChunks, StreamingMultipartEncoder, aiter_chunks
from .api import ENDPOINT_TRANSCRIBE_LIVE, _error_from_response
from .models import DictationResponse

__all__ = ["transcribe_live"]


async def transcribe_live(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    chunks: AsyncAudioChunks,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
) -> DictationResponse:
    """
    Posts a dictation request whose audio is uploaded as it arrives.

    The asyncio counterpart of `api.transcribe_live`; same endpoint, same
    chunked framing, same errors.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Dictation API base URL, e.g. `https://dictation.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        config: the JSON `config` part. None sends an empty object: the live
            endpoint requires the part ahead of the audio.
        timeout: per-operation timeout in seconds; see `api.transcribe_live`.

    Returns: the parsed dictation response.

    Raises: `DictationError` on any non-200 response — including one the
        server sends while the upload is still in flight.
    """
    encoder = StreamingMultipartEncoder()

    async def body() -> AsyncIterator[bytes]:
        # config first: the server needs sample_rate and channels before it can
        # decode a single audio byte, and rejects audio that arrives first.
        yield encoder.config_part(config) + encoder.audio_header(
            filename, audio_content_type
        )

        async for chunk in aiter_chunks(chunks):
            if chunk:
                yield chunk

        yield encoder.closing()

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE_LIVE,
        content=body(),
        headers={"Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DictationResponse.parse_obj(response.json())
