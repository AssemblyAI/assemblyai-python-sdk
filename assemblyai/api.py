from typing import BinaryIO, List, Optional
from urllib.parse import urlencode

import httpx

from . import types

ENDPOINT_TRANSCRIPT = "/v2/transcript"
ENDPOINT_UPLOAD = "/v2/upload"
ENDPOINT_LEMUR = "/lemur/v3/generate"


def _get_error_message(response: httpx.Response) -> str:
    """
    Tries to retrieve the `error` field if the response is JSON, otherwise
    returns the response text.

    Args:
        `response`: the HTTP response

    Returns: the error message
    """

    try:
        return response.json()["error"]
    except Exception:
        return response.text


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
    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to transcript url {request.audio_url}: {_get_error_message(response)}"
        )

    return types.TranscriptResponse.parse_obj(response.json())


def get_transcript(
    client: httpx.Client,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to retrieve transcript {transcript_id}: {_get_error_message(response)}",
        )

    return types.TranscriptResponse.parse_obj(response.json())


def upload_file(
    client: httpx.Client,
    audio_file: BinaryIO,
) -> str:
    """
    Uploads the given file.

    Args:
        `client`: the HTTP client
        `audio_file`: an opened file (in binary mode)

    Returns: The URL of the uploaded audio file.
    """

    response = client.post(
        ENDPOINT_UPLOAD,
        content=audio_file,
    )

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"Failed to upload audio file: {_get_error_message(response)}"
        )

    return response.json()["upload_url"]


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

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to export SRT for transcript {transcript_id}: {_get_error_message(response)}"
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

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to export VTT for transcript {transcript_id}: {_get_error_message(response)}"
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

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to search words in transcript {transcript_id}: {_get_error_message(response)}"
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
            f"redacted audio for transcript {transcript_id} is not ready yet"
        )

    if response.status_code == httpx.codes.BAD_REQUEST:
        raise types.RedactedAudioExpiredError(
            f"redacted audio for transcript {transcript_id} is no longer available"
        )

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"failed to retrieve redacted audio for transcript {transcript_id}: {_get_error_message(response)}"
        )

    return types.RedactedAudioResponse.parse_obj(response.json())


def get_sentences(
    client: httpx.Client,
    transcript_id: str,
) -> types.SentencesResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/sentences",
    )

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to retrieve sentences for transcript {transcript_id}: {_get_error_message(response)}"
        )

    return types.SentencesResponse.parse_obj(response.json())


def get_paragraphs(
    client: httpx.Client,
    transcript_id: str,
) -> types.ParagraphsResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/paragraphs",
    )

    if response.status_code != httpx.codes.ok:
        raise types.TranscriptError(
            f"failed to retrieve paragraphs for transcript {transcript_id}: {_get_error_message(response)}"
        )

    return types.ParagraphsResponse.parse_obj(response.json())


def lemur_question(
    client: httpx.Client,
    request: types.LemurQuestionRequest,
    http_timeout: Optional[float],
) -> types.LemurQuestionResponse:
    response = client.post(
        f"{ENDPOINT_LEMUR}/question-answer",
        json=request.dict(
            exclude_none=True,
        ),
        timeout=http_timeout,
    )

    if response.status_code != httpx.codes.ok:
        raise types.LemurError(
            f"failed to call Lemur questions: {_get_error_message(response)}"
        )

    return types.LemurQuestionResponse.parse_obj(response.json())


def lemur_summarize(
    client: httpx.Client,
    request: types.LemurSummaryRequest,
    http_timeout: Optional[float],
) -> types.LemurSummaryResponse:
    response = client.post(
        f"{ENDPOINT_LEMUR}/summary",
        json=request.dict(
            exclude_none=True,
        ),
        timeout=http_timeout,
    )

    if response.status_code != httpx.codes.ok:
        raise types.LemurError(
            f"failed to call Lemur summary: {_get_error_message(response)}"
        )

    return types.LemurSummaryResponse.parse_obj(response.json())


def lemur_action_items(
    client: httpx.Client,
    request: types.LemurActionItemsRequest,
    http_timeout: Optional[float],
) -> types.LemurActionItemsResponse:
    response = client.post(
        f"{ENDPOINT_LEMUR}/action-items",
        json=request.dict(
            exclude_none=True,
        ),
        timeout=http_timeout,
    )

    if response.status_code != httpx.codes.ok:
        raise types.LemurError(
            f"failed to call Lemur action items: {_get_error_message(response)}"
        )

    return types.LemurActionItemsResponse.parse_obj(response.json())


def lemur_task(
    client: httpx.Client,
    request: types.LemurTaskRequest,
    http_timeout: Optional[float],
) -> types.LemurTaskResponse:
    response = client.post(
        f"{ENDPOINT_LEMUR}/task",
        json=request.dict(
            exclude_none=True,
        ),
        timeout=http_timeout,
    )

    if response.status_code != httpx.codes.ok:
        raise types.LemurError(
            f"failed to call Lemur task: {_get_error_message(response)}"
        )

    return types.LemurTaskResponse.parse_obj(response.json())
