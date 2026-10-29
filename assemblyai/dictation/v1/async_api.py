"""The asyncio counterpart of `api.py`.

Calls the same endpoint as its sync twin and raises the same `DictationError`
through `api._error_from_response`.
"""

from typing import AsyncIterator, Callable, Optional

import httpx

from ..._multipart import (
    AsyncAudioChunks,
    StreamingMultipartEncoder,
    _Aborted,
    aiter_chunks,
)
from .api import (
    ENDPOINT_TRANSCRIBE,
    ENDPOINT_TRANSCRIBE_LIVE,
    _buffered_body,
    _error_from_response,
    _is_live_cap,
    _parse,
    _Recording,
    logger,
)
from .models import DictationResponse

__all__ = ["transcribe_buffered", "transcribe_live"]


class _AsyncRecordedChunks(_Recording):
    """
    The asyncio counterpart of `api._RecordedChunks`: the live upload's audio
    source, keeping a copy of every byte it hands out, and able to go on
    reading the source once the transport has stopped.
    """

    def __init__(self, chunks: AsyncAudioChunks) -> None:
        super().__init__()
        self._chunks = aiter_chunks(chunks)

    def __aiter__(self) -> "_AsyncRecordedChunks":
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._chunks.__anext__()
        self._keep(chunk)
        return chunk

    async def drain(self) -> None:
        """
        Reads the rest of the source into the copy, until the producer ends.

        Raises:
            DictationError: once the copy passes `BUFFERED_MAX_BYTES`; the copy
                is dropped and the producer is not read further.
            Exception: anything the producer raises, unchanged.
        """
        async for chunk in self._chunks:
            self._keep(chunk)


async def transcribe_buffered(
    client: httpx.AsyncClient,
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

    The asyncio counterpart of `api.transcribe_buffered`; same endpoint, same
    framing, same errors.
    """
    content_type, length, pieces = _buffered_body(
        audio,
        filename=filename,
        audio_content_type=audio_content_type,
        config=config,
    )

    async def body() -> AsyncIterator[bytes]:
        for piece in pieces:
            yield piece

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        content=body(),
        headers={"Content-Type": content_type, "Content-Length": str(length)},
        timeout=timeout,
    )

    return _parse(response)


async def transcribe_live(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    chunks: AsyncAudioChunks,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
    fallback_to_buffered: bool = False,
    cancelled: Optional[Callable[[], bool]] = None,
) -> DictationResponse:
    """
    Posts a dictation request whose audio is uploaded as it arrives.

    The asyncio counterpart of `api.transcribe_live`; same endpoint, same
    chunked framing, same errors, same buffered fallback.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Dictation API base URL, e.g. `https://dictation.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        config: the JSON `config` part. None sends an empty object: the live
            endpoint requires the part ahead of the audio.
        timeout: per-operation timeout in seconds; see `api.transcribe_live`.
        fallback_to_buffered: see `api.transcribe_live`.
        cancelled: see `api.transcribe_live`.

    Returns: the parsed dictation response, from the live endpoint or, after
        a fallback, from the buffered one.

    Raises: `DictationError` on any non-200 response — including one the
        server sends while the upload is still in flight. After a fallback it
        is the buffered request's own error, or a 413 `audio_too_large` raised
        here once the recording passes `BUFFERED_MAX_BYTES`.
    """
    encoder = StreamingMultipartEncoder()
    recorded = _AsyncRecordedChunks(chunks) if fallback_to_buffered else None
    source: AsyncIterator[bytes] = (
        recorded if recorded is not None else aiter_chunks(chunks)
    )

    async def body() -> AsyncIterator[bytes]:
        # config first: the server needs sample_rate and channels before it can
        # decode a single audio byte, and rejects audio that arrives first.
        yield encoder.config_part(config) + encoder.audio_header(
            filename, audio_content_type
        )

        async for chunk in source:
            if chunk:
                yield chunk

        yield encoder.closing()

    response = await client.post(
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

    # Past the live cap: read the producer until the speaker stops, then send
    # the whole recording buffered. See `api.transcribe_live`.
    await recorded.drain()
    if cancelled is not None and cancelled():
        raise _Aborted()

    audio = recorded.audio
    logger.info(
        "Dictation passed the live endpoint's 120 s cap; posting the whole "
        "recording (%d bytes) to %s",
        len(audio),
        ENDPOINT_TRANSCRIBE,
    )

    return await transcribe_buffered(
        client,
        base_url=base_url,
        audio=audio,
        filename=filename,
        audio_content_type=audio_content_type,
        config=config,
        timeout=timeout,
    )
