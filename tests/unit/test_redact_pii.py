import json
from typing import Any, Dict, Tuple

import factory
import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


def __submit_mock_request(
    httpx_mock: HTTPXMock,
    mock_response: Dict[str, Any],
    config: aai.TranscriptionConfig,
) -> Tuple[Dict[str, Any], aai.Transcript]:
    """
    Helper function to abstract mock transcriber calls with given `TranscriptionConfig`,
    and perform some common assertions.
    """

    mock_transcript_id = mock_response.get("id", "mock_id")

    # Mock initial submission response (transcript is processing)
    mock_processing_response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript",
        status_code=httpx.codes.OK,
        method="POST",
        json={
            **mock_processing_response,
            "id": mock_transcript_id,  # inject ID from main mock response
        },
    )

    # Mock polling-for-completeness response, with completed transcript
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{mock_transcript_id}",
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


def test_redact_pii_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `redact_pii` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("redact_pii") is None
    assert request_body.get("redact_pii_audio") is None
    assert request_body.get("redact_pii_policies") is None
    assert request_body.get("redact_pii_sub") is None


def test_redact_pii_enabled(httpx_mock: HTTPXMock):
    """
    Tests that enabling `redact_pii`, along with the required `redact_pii_policies`
    parameter will result in the request body containing those fields
    """
    policies = [
        aai.types.PIIRedactionPolicy.date,
        aai.types.PIIRedactionPolicy.phone_number,
    ]

    request_body, _ = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=policies,
        ),
    )

    assert request_body.get("redact_pii") is True
    assert request_body.get("redact_pii_policies") == policies


def test_redact_pii_enabled_with_optional_params(httpx_mock: HTTPXMock):
    """
    Tests that enabling `redact_pii`, along with the other optional parameters
    relevant to PII redaction, will result in the request body containing
    those fields
    """
    policies = [
        aai.types.PIIRedactionPolicy.date,
        aai.types.PIIRedactionPolicy.phone_number,
    ]
    sub_type = aai.types.PIISubstitutionPolicy.entity_name

    request_body, _ = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_audio=True,
            redact_pii_policies=policies,
            redact_pii_sub=sub_type,
        ),
    )

    assert request_body.get("redact_pii") is True
    assert request_body.get("redact_pii_audio") is True
    assert request_body.get("redact_pii_policies") == policies
    assert request_body.get("redact_pii_sub") == sub_type


def test_redact_pii_fails_without_policies(httpx_mock: HTTPXMock):
    """
    Tests that enabling `redact_pii` without specifying any policies
    will result in an exception being raised before the API call is made
    """
    with pytest.raises(ValueError) as error:
        __submit_mock_request(
            httpx_mock,
            mock_response={},
            config=aai.TranscriptionConfig(
                redact_pii=True,
                # No policies!
            ),
        )

    assert "policy" in str(error)

    # Check that the error was raised before any requests were made
    assert len(httpx_mock.get_requests()) == 0

    # Inform httpx_mock that it's okay we didn't make any requests
    httpx_mock.reset(False)


def test_redact_pii_params_excluded_when_disabled(httpx_mock: HTTPXMock):
    """
    Tests that additional PII redaction parameters are excluded from the submission
    request body if `redact_pii` itself is not enabled.
    """
    request_body, _ = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=False,
            redact_pii_audio=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_sub=aai.types.PIISubstitutionPolicy.entity_name,
        ),
    )

    assert request_body.get("redact_pii") is None
    assert request_body.get("redact_pii_audio") is None
    assert request_body.get("redact_pii_policies") is None
    assert request_body.get("redact_pii_sub") is None
