import json
from typing import Dict, Optional, Tuple

import httpx

from ... import types

# Canonical paths since the sync API gained a /v1 prefix (#18103); the
# unprefixed routes remain served for SDK versions that predate it.
ENDPOINT_TRANSCRIBE = "/v1/transcribe"
ENDPOINT_WARM = "/v1/warm"
MODEL_HEADER = "X-AAI-Model"


def _error_from_response(response: httpx.Response) -> types.SyncTranscriptError:
    """
    Builds a `SyncTranscriptError` from a non-200 response.

    The service returns an RFC 9457 problem-details envelope
    (`{"status", "title", "detail"}`); `error_code` is the snake_cased
    `title` (e.g. `"Audio Too Large"` -> `audio_too_large`). Older envelopes
    (`{"error_code", "message"}` and `{"detail"}`) are still accepted.
    """
    error_code: Optional[str] = None
    message: Optional[str] = None

    try:
        body = response.json()
        if isinstance(body, dict):
            error_code = body.get("error_code")
            title = body.get("title")
            if error_code is None and isinstance(title, str) and title:
                error_code = title.lower().replace(" ", "_")
            message = body.get("detail") or body.get("message")
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

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE,
        files=files,
        headers={MODEL_HEADER: model},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())
