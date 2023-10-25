import json
from typing import Any, Dict, Tuple

import httpx
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories


def submit_mock_transcription_request(
    httpx_mock: HTTPXMock,
    mock_response: Dict[str, Any],
    config: aai.TranscriptionConfig,
) -> Tuple[Dict[str, Any], aai.transcriber.Transcript]:
    """
    Helper function to abstract calling transcriber with given parameters,
    and perform some common assertions.

    Args:
        httpx_mock: HTTPXMock instance to use for mocking requests
        mock_response: Dict to use as mock response from API
        config: The `TranscriptionConfig` to use for transcription

    Returns:
        A tuple containing the JSON body of the initial submission request,
        and the `Transcript` object parsed from the mock response
    """

    mock_transcript_id = mock_response.get("id", "mock_id")

    # Mock initial submission response (transcript is processing)
    mock_processing_response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json={
            **mock_processing_response,
            "id": mock_transcript_id,  # inject ID from main mock response
        },
    )

    # Mock polling-for-completeness response, with completed transcript
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_transcript_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_response,
    )

    # == Make API request via SDK ==
    transcript = aai.Transcriber().transcribe(
        data="https://example.org/audio.wav",
        config=config,
    )

    # Check that submission and polling requests were made
    assert len(httpx_mock.get_requests()) == 2

    # Extract body of initial submission request
    request = httpx_mock.get_requests()[0]
    request_body = json.loads(request.content.decode())

    return request_body, transcript
