import json
from typing import Any, List, Optional, Tuple

import httpx

from ... import types
from ...sync.v1.api import _error_from_response as _shared_error_from_response
from .models import DiarizationError, DiarizationResponse

ENDPOINT_DIARIZE = "/v1/diarize"


def _error_from_response(response: httpx.Response) -> types.AssemblyAIError:
    """
    Builds a `DiarizationError` from a non-200 response.

    The Diarization API answers with the same RFC 9457 problem-details body
    (`{"status", "title", "detail"}`) as the sync API, so the parsing is
    shared and only the exception class differs.
    """

    return _shared_error_from_response(response, DiarizationError, "diarization")


def _files(
    *,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    transcript: Optional[dict],
    config: Optional[dict],
) -> List[Tuple[str, Tuple[Optional[str], Any, str]]]:
    """
    The multipart parts of a diarize request.

    Buffered, so httpx sends `Content-Length`, which the API requires (411).
    An absent `transcript` leaves its part out: segments-only mode.
    """

    files: List[Tuple[str, Tuple[Optional[str], Any, str]]] = []
    if config is not None:
        files.append(
            ("config", (None, json.dumps(config).encode(), "application/json"))
        )
    if transcript is not None:
        files.append(
            ("transcript", (None, json.dumps(transcript).encode(), "application/json"))
        )
    files.append(("audio", (filename, audio, audio_content_type)))
    return files


def _parse(response: httpx.Response) -> DiarizationResponse:
    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return DiarizationResponse.parse_obj(response.json())


def diarize(
    client: httpx.Client,
    *,
    base_url: str,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    transcript: Optional[dict],
    config: Optional[dict],
    timeout: float,
) -> DiarizationResponse:
    """
    Posts a diarize request.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Diarization API base URL, e.g. `https://diarization.assemblyai.com`.
        audio: the audio bytes.
        filename: name for the audio multipart part.
        audio_content_type: the audio part's Content-Type; selects the decoder.
        transcript: the JSON `transcript` part, or None to leave it out.
        config: the JSON `config` part, or None to leave it out.
        timeout: the httpx timeout in seconds.

    Raises: `DiarizationError` on any non-200 response.
    """

    response = client.post(
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
