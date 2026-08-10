from typing import Any, BinaryIO, Dict, List, Optional, Type, Union
from urllib.parse import urlencode

import httpx

from . import types

ENDPOINT_TRANSCRIPT = "/v2/transcript"
ENDPOINT_UPLOAD = "/v2/upload"
ENDPOINT_LEMUR_BASE = "/lemur/v3"
ENDPOINT_LEMUR = f"{ENDPOINT_LEMUR_BASE}/generate"


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
        return f"\nReason: {response.text}\nRequest: {response.request}"


def _raise_for_status(
    response: httpx.Response,
    message: str,
    error_type: Type[types.AssemblyAIError] = types.TranscriptError,
) -> None:
    """
    Raises `error_type` unless the response is a 200.

    Shared by `api` and `async_api`, so both raise the same error per endpoint.

    Args:
        `response`: the HTTP response
        `message`: what failed, e.g. `failed to retrieve transcript abc`. The
            server error is appended to it.
        `error_type`: the exception class to raise.
    """

    if response.status_code != httpx.codes.OK:
        raise error_type(
            f"{message}: {_get_error_message(response)}",
            response.status_code,
        )


def _subtitles_params(chars_per_caption: Optional[int]) -> Dict[str, Any]:
    return {"chars_per_caption": chars_per_caption} if chars_per_caption else {}


def _word_search_params(words: List[str]) -> str:
    return urlencode(
        {
            "words": ",".join(words),
        }
    )


def _list_transcripts_params(
    params: Optional[types.ListTranscriptParameters],
) -> Optional[Dict[str, Any]]:
    return (
        params.dict(
            exclude_none=True,
        )
        if params
        else None
    )


def _transcript_request_json(request: types.TranscriptRequest) -> Dict[str, Any]:
    return request.dict(
        exclude_none=True,
        by_alias=True,
    )


def _parse_redacted_audio_response(
    response: httpx.Response,
    transcript_id: str,
) -> types.RedactedAudioResponse:
    """
    Parses a redacted-audio response. Maps the 202 and 400 statuses to
    dedicated errors.

    Raises:
        RedactedAudioIncompleteError: If response indicates that the redacted audio is still processing
        RedactedAudioExpiredError: If response indicates that the redacted audio is no longer available
        TranscriptError: If we fail to get a valid response from the API at all
    """

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


def _parse_lemur_response(
    response: httpx.Response,
) -> Union[
    types.LemurStringResponse,
    types.LemurQuestionResponse,
]:
    json_data = response.json()

    if isinstance(json_data.get("response"), list):
        return types.LemurQuestionResponse.parse_obj(json_data)

    return types.LemurStringResponse.parse_obj(json_data)


def create_transcript(
    client: httpx.Client,
    request: types.TranscriptRequest,
) -> types.TranscriptResponse:
    response = client.post(
        ENDPOINT_TRANSCRIPT,
        json=_transcript_request_json(request),
    )

    _raise_for_status(response, f"failed to transcribe url {request.audio_url}")

    return types.TranscriptResponse.parse_obj(response.json())


def get_transcript(
    client: httpx.Client,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    _raise_for_status(response, f"failed to retrieve transcript {transcript_id}")

    return types.TranscriptResponse.parse_obj(response.json())


def delete_transcript(
    client: httpx.Client,
    transcript_id: str,
) -> types.TranscriptResponse:
    response = client.delete(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}",
    )

    _raise_for_status(response, f"failed to delete transcript {transcript_id}")

    return types.TranscriptResponse.parse_obj(response.json())


def upload_file(
    client: httpx.Client,
    audio_file: Union[bytes, BinaryIO],
) -> str:
    """
    Uploads the given file.

    Args:
        `client`: the HTTP client
        `audio_file`: the raw audio bytes, or an opened file (in binary mode)

    Returns: The URL of the uploaded audio file.
    """

    response = client.post(
        ENDPOINT_UPLOAD,
        content=audio_file,
    )

    _raise_for_status(response, "Failed to upload audio file")

    return response.json()["upload_url"]


def export_subtitles_srt(
    client: httpx.Client,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/srt",
        params=_subtitles_params(chars_per_caption),
    )

    _raise_for_status(response, f"failed to export SRT for transcript {transcript_id}")

    return response.text


def export_subtitles_vtt(
    client: httpx.Client,
    transcript_id: str,
    chars_per_caption: Optional[int],
) -> str:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/vtt",
        params=_subtitles_params(chars_per_caption),
    )

    _raise_for_status(response, f"failed to export VTT for transcript {transcript_id}")

    return response.text


def word_search(
    client: httpx.Client,
    transcript_id: str,
    words: List[str],
) -> types.WordSearchMatchResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/word-search",
        params=_word_search_params(words),
    )

    _raise_for_status(response, f"failed to search words in transcript {transcript_id}")

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

    return _parse_redacted_audio_response(response, transcript_id)


def get_sentences(
    client: httpx.Client,
    transcript_id: str,
) -> types.SentencesResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/sentences",
    )

    _raise_for_status(
        response, f"failed to retrieve sentences for transcript {transcript_id}"
    )

    return types.SentencesResponse.parse_obj(response.json())


def get_paragraphs(
    client: httpx.Client,
    transcript_id: str,
) -> types.ParagraphsResponse:
    response = client.get(
        f"{ENDPOINT_TRANSCRIPT}/{transcript_id}/paragraphs",
    )

    _raise_for_status(
        response, f"failed to retrieve paragraphs for transcript {transcript_id}"
    )

    return types.ParagraphsResponse.parse_obj(response.json())


def list_transcripts(
    client: httpx.Client,
    params: Optional[types.ListTranscriptParameters],
) -> types.ListTranscriptResponse:
    response = client.get(
        ENDPOINT_TRANSCRIPT,
        params=_list_transcripts_params(params),
    )

    _raise_for_status(response, "failed to retrieve transcripts", types.AssemblyAIError)

    return types.ListTranscriptResponse.parse_obj(response.json())


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

    _raise_for_status(response, "failed to call Lemur questions", types.LemurError)

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

    _raise_for_status(response, "failed to call Lemur summary", types.LemurError)

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

    _raise_for_status(response, "failed to call Lemur action items", types.LemurError)

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

    _raise_for_status(response, "failed to call Lemur task", types.LemurError)

    return types.LemurTaskResponse.parse_obj(response.json())


def lemur_purge_request_data(
    client: httpx.Client,
    request: types.LemurPurgeRequest,
    http_timeout: Optional[float],
) -> types.LemurPurgeResponse:
    response = client.delete(
        f"{ENDPOINT_LEMUR_BASE}/{request.request_id}",
        timeout=http_timeout,
    )

    _raise_for_status(
        response,
        f"Failed to purge LeMUR request data for provided request ID: {request.request_id}. Error",
        types.LemurError,
    )

    return types.LemurPurgeResponse.parse_obj(response.json())


def lemur_get_response_data(
    client: httpx.Client,
    request_id: str,
    http_timeout: Optional[float],
) -> Union[
    types.LemurStringResponse,
    types.LemurQuestionResponse,
]:
    response = client.get(
        f"{ENDPOINT_LEMUR_BASE}/{request_id}",
        timeout=http_timeout,
    )

    _raise_for_status(
        response,
        f"Failed to get LeMUR response data for provided request ID: {request_id}. Error",
        types.LemurError,
    )

    return _parse_lemur_response(response)
