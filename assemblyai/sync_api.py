import json
from typing import Optional

import httpx

from . import types

ENDPOINT_TRANSCRIBE = "/transcribe"
MODEL_HEADER = "X-AAI-Model"


def _error_from_response(response: httpx.Response) -> types.SyncTranscriptError:
    """
    Builds a `SyncTranscriptError` from a non-200 response.

    The service uses two error envelopes: `{"error_code", "message"}` for
    audio/capacity/inference errors and `{"detail"}` for auth and rate-limit
    errors. Parse by status code, not by assuming `error_code` is present.
    """
    error_code: Optional[str] = None
    message: Optional[str] = None

    try:
        body = response.json()
        if isinstance(body, dict):
            error_code = body.get("error_code")
            message = body.get("message") or body.get("detail")
    except Exception:
        message = response.text or None

    if not message:
        message = f"sync transcription failed with status {response.status_code}"

    retry_after_header = response.headers.get("retry-after")
    retry_after = (
        int(retry_after_header)
        if retry_after_header and retry_after_header.isdigit()
        else None
    )

    return types.SyncTranscriptError(
        message,
        status_code=response.status_code,
        error_code=error_code,
        retry_after=retry_after,
    )


def transcribe(
    client: httpx.Client,
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
    files = {"audio": (filename, audio, audio_content_type)}
    if config:
        files["config"] = (None, json.dumps(config), "application/json")

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        files=files,
        headers={MODEL_HEADER: model},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())
