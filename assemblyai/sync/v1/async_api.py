"""The asyncio counterpart of `api.py`.

Calls the same endpoint as its sync twin and raises the same
`SyncTranscriptError` through `api._error_from_response`.
"""

import json
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx

from ... import types
from ._multipart import AsyncAudioChunks, StreamingMultipartEncoder, aiter_chunks
from .api import (
    ENDPOINT_TRANSCRIBE,
    ENDPOINT_TRANSCRIBE_STREAM,
    MODEL_HEADER,
    _error_from_response,
)

__all__ = ["transcribe", "transcribe_stream"]


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
    Posts a single synchronous transcription request.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the sync API base URL, e.g. `https://sync.assemblyai.com`.
        audio: raw audio bytes (WAV container or S16LE PCM).
        filename: name for the audio multipart part.
        audio_content_type: `audio/wav` or `audio/pcm`; selects the decoder.
        model: sent as the `X-AAI-Model` routing header.
        config: the JSON `config` part, or None to omit it.
        timeout: per-request timeout in seconds.

    Returns: the parsed transcript response.

    Raises: `SyncTranscriptError` on any non-200 response.
    """
    files: Dict[str, Tuple[Optional[str], bytes, str]] = {
        "audio": (filename, audio, audio_content_type)
    }
    if config:
        # httpx <0.23 rejects a `str` multipart part; encode to bytes so the
        # config part works across the full supported httpx range (>=0.19).
        files["config"] = (
            None,
            json.dumps(config).encode("utf-8"),
            "application/json",
        )

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        files=files,
        headers={MODEL_HEADER: model},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())


async def transcribe_stream(
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

    The asyncio counterpart of `api.transcribe_stream`; same endpoint, same
    chunked framing, same errors.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the sync API base URL, e.g. `https://sync.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: `audio/wav` or `audio/pcm`; selects the decoder.
        model: sent as the `X-AAI-Model` routing header.
        config: the JSON `config` part, or None to omit it.
        timeout: per-operation timeout in seconds; see `api.transcribe_stream`.

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
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE_STREAM,
        content=body(),
        headers={MODEL_HEADER: model, "Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())
