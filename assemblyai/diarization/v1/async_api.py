"""The asyncio counterpart of `api.py`: same endpoint, parts and errors."""

from typing import Optional

import httpx

from .api import ENDPOINT_DIARIZE, _files, _parse
from .models import DiarizationResponse

__all__ = ["diarize"]


async def diarize(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    transcript: Optional[dict],
    config: Optional[dict],
    timeout: float,
) -> DiarizationResponse:
    """Posts a diarize request. The asyncio counterpart of `api.diarize`."""

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_DIARIZE,
        files=_files(
            audio=audio,
            filename=filename,
            audio_content_type=audio_content_type,
            transcript=transcript,
            config=config,
        ),
        timeout=timeout,
    )
    return _parse(response)
