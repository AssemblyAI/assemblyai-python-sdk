import json
from typing import Dict, Optional, Tuple

import httpx

from ... import types
from ...sync.v1.api import _error_from_response as _shared_error_from_response
from .models import DictationError, DictationResponse

ENDPOINT_TRANSCRIBE = "/v1/transcribe"
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


def transcribe(
    client: httpx.Client,
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

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        files=files,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DictationResponse.parse_obj(response.json())
