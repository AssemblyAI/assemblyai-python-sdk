"""Asyncio counterparts of the request functions in `api`.

Each function calls the same endpoint as its sync twin, and raises the same
exception type and message through `_base._raise_for_status`.
"""

from typing import AsyncIterable, Dict, List, Optional, Union
from urllib.parse import urlencode

import httpx

from ... import types
from ...api import ENDPOINT_UPLOAD
from ._base import _raise_for_status
from .api import ENDPOINT_TRANSCRIPT

__all__ = [
    "create_transcript",
    "delete_transcript",
    "export_subtitles_srt",
    "export_subtitles_vtt",
    "get_paragraphs",
    "get_redacted_audio",
    "get_sentences",
    "get_transcript",
    "list_transcripts",
    "upload_file",
    "word_search",
]


async def create_transcript(
    client: httpx.AsyncClient,
    request: types.TranscriptRequest,
) -> types.TranscriptResponse:
    response = await client.post(
        ENDPOINT_TRANSCRIPT,
        json=request.dict(
            exclude_none=True,
            by_alias=True,
        ),
    )

    _raise_for_status(response, f"failed to transcribe url {request.audio_url}")

    return types.TranscriptResponse.parse_obj(response.json())


async def get_transcript(
    client: httpx.AsyncClient,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    _raise_for_status(response, f"failed to retrieve transcript {transcript_id}")

    return types.TranscriptResponse.parse_obj(response.json())


async def delete_transcript(
    client: httpx.AsyncClient,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = await client.delete(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    _raise_for_status(response, f"failed to delete transcript {transcript_id}")

    return types.TranscriptResponse.parse_obj(response.json())


async def upload_file(
    client: httpx.AsyncClient,
    audio_file: Union[bytes, AsyncIterable[bytes]],
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Uploads the given audio.

    Args:
        `client`: the HTTP client
        `audio_file`: the raw audio bytes, or an async iterable of bytes. Do
            not pass a blocking file object. httpx would read it on the event
            loop. See `AsyncTranscriber.upload_file`.
        `headers`: extra headers for the body, such as `Content-Length`. Without
            it, httpx uses chunked transfer encoding for an async iterable.

    Returns: The URL of the uploaded audio file.
    """

    response = await client.post(
        ENDPOINT_UPLOAD,
        content=audio_file,
        headers=headers,
    )

    _raise_for_status(response, "Failed to upload audio file")

    return response.json()["upload_url"]


async def export_subtitles_srt(
    client: httpx.AsyncClient,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/srt",
        params={"chars_per_caption": chars_per_caption} if chars_per_caption else {},
    )

    _raise_for_status(response, f"failed to export SRT for transcript {transcript_id}")

    return response.text


async def export_subtitles_vtt(
    client: httpx.AsyncClient,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/vtt",
        params={"chars_per_caption": chars_per_caption} if chars_per_caption else {},
    )

    _raise_for_status(response, f"failed to export VTT for transcript {transcript_id}")

    return response.text


async def word_search(
    client: httpx.AsyncClient,
    transcript_id: str,
    words: List[str],
) -> types.WordSearchMatchResponse:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/word-search",
        params=urlencode(
            {
                "words": ",".join(words),
            }
        ),
    )

    _raise_for_status(response, f"failed to search words in transcript {transcript_id}")

    return types.WordSearchMatchResponse.parse_obj(response.json())


async def get_redacted_audio(
    client: httpx.AsyncClient,
    transcript_id: str,
) -> types.RedactedAudioResponse:
    """
    Retrieves the object containing the redacted audio URL for the given transcript.

    Raises:
        RedactedAudioIncompleteError: If response indicates that the redacted audio is still processing
        RedactedAudioExpiredError: If response indicates that the redacted audio is no longer available
        TranscriptError: If we fail to get a valid response from the API at all

    Returns:
        `RedactedAudioResponse`, which contains the URL of the redacted audio
    """

    response = await client.get(f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/redacted-audio")

    if response.status_code == httpx.codes.ACCEPTED:
        raise types.RedactedAudioIncompleteError(
            f"redacted audio for transcript {transcript_id} is not ready yet",
            response.status_code,
        )

    if response.status_code == httpx.codes.BAD_REQUEST:
        raise types.RedactedAudioExpiredError(
            f"redacted audio for transcript {transcript_id} is no longer available",
            response.status_code,
        )

    _raise_for_status(
        response, f"failed to retrieve redacted audio for transcript {transcript_id}"
    )

    return types.RedactedAudioResponse.parse_obj(response.json())


async def get_sentences(
    client: httpx.AsyncClient,
    transcript_id: str,
) -> types.SentencesResponse:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/sentences",
    )

    _raise_for_status(
        response, f"failed to retrieve sentences for transcript {transcript_id}"
    )

    return types.SentencesResponse.parse_obj(response.json())


async def get_paragraphs(
    client: httpx.AsyncClient,
    transcript_id: str,
) -> types.ParagraphsResponse:
    response = await client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/paragraphs",
    )

    _raise_for_status(
        response, f"failed to retrieve paragraphs for transcript {transcript_id}"
    )

    return types.ParagraphsResponse.parse_obj(response.json())


async def list_transcripts(
    client: httpx.AsyncClient,
    params: Optional[types.ListTranscriptParameters],
) -> types.ListTranscriptResponse:
    response = await client.get(
        ENDPOINT_TRANSCRIPT,
        params=(
            params.dict(
                exclude_none=True,
            )
            if params
            else None
        ),
    )

    _raise_for_status(response, "failed to retrieve transcripts", types.AssemblyAIError)

    return types.ListTranscriptResponse.parse_obj(response.json())
