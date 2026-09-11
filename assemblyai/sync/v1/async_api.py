"""The asyncio counterpart of `api.py`.

Calls the same endpoint as its sync twin and raises the same
`SyncTranscriptError` through `api._error_from_response`.
"""

from typing import AsyncIterator, Optional

import httpx

from ... import types
from ..._multipart import AsyncAudioChunks, StreamingMultipartEncoder, aiter_chunks
from .api import (
    ENDPOINT_TRANSCRIBE_LIVE,
    MODEL_HEADER,
    _error_from_response,
)

__all__ = ["transcribe", "transcribe_live"]


async def transcribe(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    model: str,
    config: Optional[dict],
    timeout: float,
) -> types.SyncTranscriptResponse:
    """
    Posts a transcription request for audio that is already complete.

    The asyncio counterpart of `api.transcribe`: the same single-chunk send
    over the live connection, which is the only one this client opens.
    """
    return await transcribe_live(
        client,
        base_url=base_url,
        chunks=(audio,),
        filename=filename,
        audio_content_type=audio_content_type,
        model=model,
        config=config,
        timeout=timeout,
    )


async def transcribe_live(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    chunks: AsyncAudioChunks,
    filename: str,
    audio_content_type: str,
    model: str,
    config: Optional[dict],
    timeout: float,
) -> types.SyncTranscriptResponse:
    """
    Posts a transcription request whose audio is uploaded as it arrives.

    The asyncio counterpart of `api.transcribe_live`; same endpoint, same
    chunked framing, same errors.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the sync API base URL, e.g. `https://sync.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: `audio/wav` or `audio/pcm`; selects the decoder.
        model: sent as the `X-AAI-Model` routing header.
        config: the JSON `config` part. None sends an empty object: the
            streaming endpoint requires the part ahead of the audio.
        timeout: per-operation timeout in seconds; see `api.transcribe_live`.

    Returns: the parsed transcript response.

    Raises: `SyncTranscriptError` on any non-200 response — including one the
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
        headers={MODEL_HEADER: model, "Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())
