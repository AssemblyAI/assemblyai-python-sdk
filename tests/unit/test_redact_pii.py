import json
from typing import Any, Dict, Tuple

import httpx
import pytest
from pytest_httpx import HTTPXMock
from pytest_mock import MockerFixture

import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class TranscriptWithPIIRedactionResponseFactory(
    factories.TranscriptCompletedResponseFactory
):
    redact_pii = True
    redact_pii_audio = True
    redact_pii_policies = [
        aai.types.PIIRedactionPolicy.date,
    ]


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
    request_body, _ = __submit_mock_request(
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
            TranscriptWithPIIRedactionResponseFactory
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
            TranscriptWithPIIRedactionResponseFactory
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


def __get_redacted_audio_api_url(transcript: aai.Transcript) -> str:
    return f"{aai.settings.base_url}/transcript/{transcript.id}/redacted-audio"


REDACTED_AUDIO_URL = "https://example.org/redacted-audio.wav"


def __mock_successful_pii_audio_responses(
    httpx_mock: HTTPXMock, transcript: aai.Transcript
):
    # Mock pending redacted audio response on first call
    httpx_mock.add_response(
        url=__get_redacted_audio_api_url(transcript),
        status_code=202,
        method="GET",
        json={},
    )

    # Mock completed redacted audio response on second call
    httpx_mock.add_response(
        url=__get_redacted_audio_api_url(transcript),
        status_code=httpx.codes.OK,
        method="GET",
        json={
            "redacted_audio_url": REDACTED_AUDIO_URL,
            "status": "redacted_audio_ready",
        },
    )


def __mock_failed_pii_audio_responses(
    httpx_mock: HTTPXMock, transcript: aai.Transcript
):
    httpx_mock.add_response(
        url=__get_redacted_audio_api_url(transcript),
        status_code=400,
        method="GET",
        json={},
    )


def test_get_pii_redacted_audio_url(httpx_mock: HTTPXMock):
    """
    Tests that the PII-redacted audio URL can be retrieved from the API
    with a successful response
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            TranscriptWithPIIRedactionResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_audio=True,
        ),
    )

    __mock_successful_pii_audio_responses(httpx_mock, transcript)
    redacted_audio_url = transcript.get_redacted_audio_url()

    # Ensure we made a third and fourth network request to get the redacted audio information
    assert len(httpx_mock.get_requests()) == 4

    assert redacted_audio_url == REDACTED_AUDIO_URL


def test_get_pii_redacted_audio_url_fails_if_redact_pii_not_enabled_for_transcript(
    httpx_mock: HTTPXMock,
):
    """
    Tests that an error is thrown before any requests are made if
    `redact_pii` was not enabled for the transcript and
    `get_redacted_audio_url` is called
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory  # standard response
        )(),
        config=aai.TranscriptionConfig(),  # blank config
    )

    with pytest.raises(ValueError) as error:
        transcript.get_redacted_audio_url()

    assert "redact_pii" in str(error)

    # Ensure we never made the additional requests to get the redacted audio information
    assert len(httpx_mock.get_requests()) == 2


def test_get_pii_redacted_audio_url_fails_if_redact_pii_audio_not_enabled_for_transcript(
    httpx_mock: HTTPXMock,
):
    """
    Tests that an error is thrown before any requests are made if
    `redact_pii_audio` was not enabled for the transcript and
    `get_redacted_audio_url` is called
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response={
            **factories.generate_dict_factory(
                TranscriptWithPIIRedactionResponseFactory
            )(),
            "redact_pii_audio": False,
        },
        config=aai.TranscriptionConfig(
            redact_pii=True, redact_pii_policies=[aai.types.PIIRedactionPolicy.date]
        ),
    )

    with pytest.raises(ValueError) as error:
        transcript.get_redacted_audio_url()

    assert "redact_pii_audio" in str(error)

    # Ensure we never made the additional requests to get the redacted audio information
    assert len(httpx_mock.get_requests()) == 2


def test_get_pii_redacted_audio_url_fails_if_bad_response(httpx_mock: HTTPXMock):
    """
    Tests that `get_redacted_audio_url` raises a `RedactedAudioUnavailableError` if
    the request to fetch the redacted audio URL returns a `400` status code, indicating
    that the redacted audio has expired
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            TranscriptWithPIIRedactionResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_audio=True,
        ),
    )

    __mock_failed_pii_audio_responses(httpx_mock, transcript)
    with pytest.raises(aai.types.RedactedAudioExpiredError):
        transcript.get_redacted_audio_url()


def test_save_pii_redacted_audio(httpx_mock: HTTPXMock, mocker: MockerFixture):
    """
    Tests that calling `save_redacted_audio` will download the redacted audio file
    to the caller's file system
    """

    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            TranscriptWithPIIRedactionResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_audio=True,
        ),
    )

    # Mock response that returns the redacted-audio URL
    __mock_successful_pii_audio_responses(httpx_mock, transcript)

    # Mock the redacted-audio URL response
    mock_audio_file_bytes = b"pretend this is a WAV file"
    httpx_mock.add_response(
        url=REDACTED_AUDIO_URL,
        status_code=httpx.codes.OK,
        method="GET",
        content=mock_audio_file_bytes,
    )

    # Set up mocks for writing to disk
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    # Download the file
    downloaded_filepath = "redacted_audio.wav"
    transcript.save_redacted_audio(downloaded_filepath)

    # Ensure correct filepath was written to
    mock_file.assert_called_once_with(downloaded_filepath, "wb")

    # Ensure correct file content was written
    write_calls = mock_file().write.call_args_list
    full_written_bytes = b"".join(call.args[0] for call in write_calls)
    assert full_written_bytes == mock_audio_file_bytes


def test_save_pii_redacted_audio_fails_if_redact_pii_not_enabled_for_transcript(
    httpx_mock: HTTPXMock,
):
    """
    Tests that an error is thrown before any requests are made if
    `redact_pii` was not enabled for the transcript and
    `save_redacted_audio` is called
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory  # standard response
        )(),
        config=aai.TranscriptionConfig(),  # blank config
    )

    with pytest.raises(ValueError) as error:
        transcript.save_redacted_audio("redacted_audio.wav")

    assert "redact_pii" in str(error)

    # Ensure we never made the additional requests to get the redacted audio information
    assert len(httpx_mock.get_requests()) == 2


def test_save_pii_redacted_audio_fails_if_redact_pii_audio_not_enabled_for_transcript(
    httpx_mock: HTTPXMock,
):
    """
    Tests that an error is thrown before any requests are made if
    `redact_pii_audio` was not enabled for the transcript and
    `get_redacted_audio_url` is called
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response={
            **factories.generate_dict_factory(
                TranscriptWithPIIRedactionResponseFactory
            )(),
            "redact_pii_audio": False,
        },
        config=aai.TranscriptionConfig(
            redact_pii=True, redact_pii_policies=[aai.types.PIIRedactionPolicy.date]
        ),
    )

    with pytest.raises(ValueError) as error:
        transcript.save_redacted_audio("redacted_audio.wav")

    assert "redact_pii_audio" in str(error)

    # Ensure we never made the additional requests to get the redacted audio information
    assert len(httpx_mock.get_requests()) == 2


def test_save_pii_redacted_audio_fails_if_bad_response(httpx_mock: HTTPXMock):
    """
    Tests that `save_redacted_audio` raises a `RedactedAudioUnavailableError` if
    the request to fetch the redacted audio URL returns a `400` status code,
    indicating that the redacted audio has expired
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            TranscriptWithPIIRedactionResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_audio=True,
        ),
    )

    __mock_failed_pii_audio_responses(httpx_mock, transcript)
    with pytest.raises(aai.types.RedactedAudioExpiredError):
        transcript.save_redacted_audio("redacted_audio.wav")


def test_save_pii_redacted_audio_fails_if_bad_audio_url_response(httpx_mock: HTTPXMock):
    """
    Tests that `save_redacted_audio` raises a `RedactedAudioUnavailableError` if
    the request to fetch the redacted audio **file** returns a non-200 status code
    """
    _, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            TranscriptWithPIIRedactionResponseFactory
        )(),
        config=aai.TranscriptionConfig(
            redact_pii=True,
            redact_pii_policies=[aai.types.PIIRedactionPolicy.date],
            redact_pii_audio=True,
        ),
    )

    __mock_successful_pii_audio_responses(httpx_mock, transcript)
    httpx_mock.add_response(
        url=REDACTED_AUDIO_URL,
        status_code=httpx.codes.NOT_FOUND,
        method="GET",
    )
    with pytest.raises(aai.types.RedactedAudioUnavailableError):
        transcript.save_redacted_audio("redacted_audio.wav")
