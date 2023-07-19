import json
from typing import Any, Dict

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories

aai.settings.api_key = "test"


def __submit_request(httpx_mock: HTTPXMock, **params) -> Dict[str, Any]:
    """
    Helper function to abstract calling transcriber with given parameters,
    and perform some common assertions.

    Returns the body (dictionary) of the initial submission request.
    """
    summary = "example summary"

    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # Mock initial submission response
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_transcript_response,
    )

    # Mock polling-for-completeness response, with mock summary result
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_transcript_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json={**mock_transcript_response, "summary": summary},
    )

    # == Make API request via SDK ==
    transcript = aai.Transcriber().transcribe(
        data="https://example.org/audio.wav",
        config=aai.TranscriptionConfig(
            **params,
        ),
    )

    # Check that submission and polling requests were made
    assert len(httpx_mock.get_requests()) == 2

    # Check that summary field from response was traced back through SDK classes
    assert transcript.summary == summary

    # Extract and return body of initial submission request
    request = httpx_mock.get_requests()[0]
    return json.loads(request.content.decode())


@pytest.mark.parametrize("required_field", ["punctuate", "format_text"])
def test_summarization_fails_without_required_field(
    httpx_mock: HTTPXMock, required_field: str
):
    """
    Tests whether the SDK raises an error before making a request
    if `summarization` is enabled and the given required field is disabled
    """
    with pytest.raises(ValueError) as error:
        __submit_request(httpx_mock, summarization=True, **{required_field: False})

    # Check that the error message informs the user of the invalid parameter
    assert required_field in str(error)

    # Check that the error was raised before any requests were made
    assert len(httpx_mock.get_requests()) == 0

    # Inform httpx_mock that it's okay we didn't make any requests
    httpx_mock.reset(False)


def test_summarization_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `summarization` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body = __submit_request(httpx_mock)
    assert request_body.get("summarization") is None


def test_default_summarization_params(httpx_mock: HTTPXMock):
    """
    Tests that including `summarization=True` in the `TranscriptionConfig`
    will result in `summarization=True` in the request body.
    """
    request_body = __submit_request(httpx_mock, summarization=True)
    assert request_body.get("summarization") == True


def test_summarization_with_params(httpx_mock: HTTPXMock):
    """
    Tests that including additional summarization parameters along with
    `summarization=True` in the `TranscriptionConfig` will result in all
    parameters being included in the request as well.
    """

    summary_model = aai.SummarizationModel.conversational
    summary_type = aai.SummarizationType.bullets

    request_body = __submit_request(
        httpx_mock,
        summarization=True,
        summary_model=summary_model,
        summary_type=summary_type,
    )

    assert request_body.get("summarization") == True
    assert request_body.get("summary_model") == summary_model
    assert request_body.get("summary_type") == summary_type


def test_summarization_params_excluded_when_disabled(httpx_mock: HTTPXMock):
    """
    Tests that additional summarization parameters are excluded from the submission
    request body if `summarization` itself is not enabled.
    """
    request_body = __submit_request(
        httpx_mock,
        summarization=False,
        summary_model=aai.SummarizationModel.conversational,
        summary_type=aai.SummarizationType.bullets,
    )

    assert request_body.get("summarization") is None
    assert request_body.get("summary_model") is None
    assert request_body.get("summary_type") is None
