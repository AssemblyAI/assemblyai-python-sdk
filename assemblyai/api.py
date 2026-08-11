from typing import BinaryIO, Optional, Union

import httpx

from . import types

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

    if response.status_code != httpx.codes.OK:
        raise types.TranscriptError(
            f"Failed to upload audio file: {_get_error_message(response)}",
            response.status_code,
        )

    return response.json()["upload_url"]


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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"failed to call Lemur questions: {_get_error_message(response)}",
            response.status_code,
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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"failed to call Lemur summary: {_get_error_message(response)}",
            response.status_code,
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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"failed to call Lemur action items: {_get_error_message(response)}",
            response.status_code,
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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"failed to call Lemur task: {_get_error_message(response)}",
            response.status_code,
        )

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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"Failed to purge LeMUR request data for provided request ID: {request.request_id}. Error: {_get_error_message(response)}",
            response.status_code,
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

    if response.status_code != httpx.codes.OK:
        raise types.LemurError(
            f"Failed to get LeMUR response data for provided request ID: {request_id}. Error: {_get_error_message(response)}",
            response.status_code,
        )

    json_data = response.json()

    if isinstance(json_data.get("response"), list):
        return types.LemurQuestionResponse.parse_obj(json_data)

    return types.LemurStringResponse.parse_obj(json_data)


# Canonical location for the prerecorded transcript endpoints is
# ``assemblyai.prerecorded.v2.api``.
from .prerecorded.v2.api import (  # noqa: E402, F401
    ENDPOINT_TRANSCRIPT,
    create_transcript,
    delete_transcript,
    export_subtitles_srt,
    export_subtitles_vtt,
    get_paragraphs,
    get_redacted_audio,
    get_sentences,
    get_transcript,
    list_transcripts,
    word_search,
)
