"""The asyncio counterpart of `api.py`.

Calls the same endpoint as its sync twin and raises the same `DictationError`
through `api._error_from_response`.
"""

import json
from typing import Dict, Optional, Tuple

import httpx

from .api import ENDPOINT_TRANSCRIBE, _error_from_response
from .models import DictationResponse

__all__ = ["transcribe"]


async def transcribe(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    config: Optional[dict],
    timeout: float,
) -> DictationResponse:
    """
    Posts a single dictation transcription request.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Dictation API base URL, e.g. `https://dictation.assemblyai.com`.
        audio: raw audio bytes (a container format or S16LE PCM).
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        config: the JSON `config` part, or None to omit it.
        timeout: per-request timeout in seconds.

    Returns: the parsed dictation response.

    Raises: `DictationError` on any non-200 response.
    """
    files: Dict[str, Tuple[Optional[str], bytes, str]] = {}
    if config:
        # The server reads the config part before the audio stream, so it must
        # come first in the multipart body — httpx preserves this insertion
        # order. httpx <0.23 rejects a `str` part; encode to bytes so the
        # config part works across the full supported httpx range (>=0.19).
        files["config"] = (
            None,
            json.dumps(config).encode("utf-8"),
            "application/json",
        )
    files["audio"] = (filename, audio, audio_content_type)

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        files=files,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DictationResponse.parse_obj(response.json())
