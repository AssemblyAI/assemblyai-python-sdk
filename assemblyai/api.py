from typing import BinaryIO, Union

import httpx

from . import types

ENDPOINT_UPLOAD = "/v2/upload"


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
