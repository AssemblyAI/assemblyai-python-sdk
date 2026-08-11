"""HTTP calls against the v2 transcript API endpoints."""

from typing import List, Optional
from urllib.parse import urlencode

import httpx

from ... import api as _root_api
from ... import types

ENDPOINT_TRANSCRIPT = "/v2/transcript"


def create_transcript(
    client: httpx.Client,
    request: types.TranscriptRequest,
) -> types.TranscriptResponse:
    response = client.post(
        ENDPOINT_TRANSCRIPT,
        json=request.dict(
            exclude_none=True,
            by_alias=True,
        ),
    )
    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to transcribe url {request.audio_url}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.TranscriptResponse.parse_obj(response.json())


def get_transcript(
    client: httpx.Client,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to retrieve transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.TranscriptResponse.parse_obj(response.json())


def delete_transcript(
    client: httpx.Client,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = client.delete(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to delete transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.TranscriptResponse.parse_obj(response.json())


def export_subtitles_srt(
    client: httpx.Client,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    params = {}

    if chars_per_caption:
        params = {
            "chars_per_caption": chars_per_caption,
        }

    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/srt",
        params=params,
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to export SRT for transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return response.text


def export_subtitles_vtt(
    client: httpx.Client,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    params = {}

    if chars_per_caption:
        params = {
            "chars_per_caption": chars_per_caption,
        }

    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/vtt",
        params=params,
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to export VTT for transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return response.text


def word_search(
    client: httpx.Client,
    transcript_id: str,
    words: List[str],
) -> types.WordSearchMatchResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/word-search",
        params=urlencode(
            {
                "words": ",".join(words),
            }
        ),
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to search words in transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.WordSearchMatchResponse.parse_obj(response.json())


def get_redacted_audio(
    client: httpx.Client, transcript_id: str
) -> types.RedactedAudioResponse:
    """
    Retrieves the object containing the redacted audio URL for the given transcript.

    Raises:
        RedactedAudioIncompleteError: If response indicates that the redacted audio is still processing
        RedactedAudioUnavailableError: If response indicates that the redacted audio is not available
        TranscriptError: If we fail to get a valid response from the API at all

    Returns:
        `RedactedAudioResponse`, which contains the URL of the redacted audio
    """

    response = client.get(f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/redacted-audio")

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

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to retrieve redacted audio for transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.RedactedAudioResponse.parse_obj(response.json())


def get_sentences(
    client: httpx.Client,
    transcript_id: str,
) -> types.SentencesResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/sentences",
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to retrieve sentences for transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.SentencesResponse.parse_obj(response.json())


def get_paragraphs(
    client: httpx.Client,
    transcript_id: str,
) -> types.ParagraphsResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/paragraphs",
    )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to retrieve paragraphs for transcript {transcript_id}: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.ParagraphsResponse.parse_obj(response.json())


def list_transcripts(
    client: httpx.Client,
    params: Optional[types.ListTranscriptParameters],
) -> types.ListTranscriptResponse:
    response = client.get(
        ENDPOINT_TRANSCRIPT,
        params=(
            params.dict(
                exclude_none=True,
            )
            if params
            else None
        ),
    )

    if response.status_code != httpx.codes.OK:
        raise types.AssemblyAIError(
            f"failed to retrieve transcripts: {_root_api._get_error_message(response)}",
            response.status_code,
        )

    return types.ListTranscriptResponse.parse_obj(response.json())
