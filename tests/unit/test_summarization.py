import factory
import pytest
from pytest_httpx import HTTPXMock

import tests.unit.factories as factories
import tests.unit.unit_test_utils as test_utils
import assemblyai as aai

aai.settings.api_key = "test"


class SummarizationResponseFactory(factories.TranscriptCompletedResponseFactory):
    summary = factory.Faker("sentence")


@pytest.mark.parametrize("required_field", ["punctuate", "format_text"])
def test_summarization_fails_without_required_field(
    httpx_mock: HTTPXMock, required_field: str
):
    """
    Tests whether the SDK raises an error before making a request
    if `summarization` is enabled and the given required field is disabled
    """
    with pytest.raises(ValueError) as error:
        test_utils.submit_mock_transcription_request(
            httpx_mock,
            {},
            config=aai.TranscriptionConfig(
                summarization=True,
                **{required_field: False},  # type: ignore
            ),
        )

    # Check that the error message informs the user of the invalid parameter
    assert required_field in str(error)

    # Check that the error was raised before any requests were made
    assert len(httpx_mock.get_requests()) == 0


def test_summarization_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `summarization` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    mock_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response,
        config=aai.TranscriptionConfig(),
    )

    # Check that request body was properly defined
    assert request_body.get("summarization") is None

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.summary is None


def test_default_summarization_params(httpx_mock: HTTPXMock):
    """
    Tests that including `summarization=True` in the `TranscriptionConfig`
    will result in `summarization=True` in the request body.
    """
    mock_response = factories.generate_dict_factory(SummarizationResponseFactory)()
    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock, mock_response, aai.TranscriptionConfig(summarization=True)
    )

    # Check that request body was properly defined
    assert request_body.get("summarization") is True
    assert request_body.get("summary_model") is None
    assert request_body.get("summary_type") is None

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.summary == mock_response["summary"]


def test_summarization_with_params(httpx_mock: HTTPXMock):
    """
    Tests that including additional summarization parameters along with
    `summarization=True` in the `TranscriptionConfig` will result in all
    parameters being included in the request as well.
    """

    summary_model = aai.SummarizationModel.conversational
    summary_type = aai.SummarizationType.bullets

    mock_response = factories.generate_dict_factory(SummarizationResponseFactory)()

    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response,
        aai.TranscriptionConfig(
            summarization=True,
            summary_model=summary_model,
            summary_type=summary_type,
        ),
    )

    # Check that request body was properly defined
    assert request_body.get("summarization") is True
    assert request_body.get("summary_model") == summary_model
    assert request_body.get("summary_type") == summary_type

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.summary == mock_response["summary"]


def test_summarization_params_excluded_when_disabled(httpx_mock: HTTPXMock):
    """
    Tests that additional summarization parameters are excluded from the submission
    request body if `summarization` itself is not enabled.
    """
    mock_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response,
        aai.TranscriptionConfig(
            summarization=False,
            summary_model=aai.SummarizationModel.conversational,
            summary_type=aai.SummarizationType.bullets,
        ),
    )

    # Check that request body was properly defined
    assert request_body.get("summarization") is None
    assert request_body.get("summary_model") is None
    assert request_body.get("summary_type") is None

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.summary is None


@pytest.mark.parametrize(
    "speaker_separation_param",
    [
        {"multichannel": True},
        {"speaker_labels": True},
        {"dual_channel": True},
    ],
    ids=["multichannel", "speaker_labels", "dual_channel"],
)
def test_conversational_summary_with_speaker_separation(
    httpx_mock: HTTPXMock, speaker_separation_param: dict
):
    """
    Tests that `summary_model=conversational` is accepted together with any of
    the valid speaker-separation flags (`multichannel`, `speaker_labels`,
    `dual_channel`) without a client-side error.

    Regression test for https://github.com/AssemblyAI/assemblyai-node-sdk/issues/71 —
    `multichannel=True` was previously undocumented as a valid alternative to
    `speaker_labels`/`dual_channel` for conversational summaries.
    """
    mock_response = factories.generate_dict_factory(SummarizationResponseFactory)()
    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response,
        aai.TranscriptionConfig(
            summarization=True,
            summary_model=aai.SummarizationModel.conversational,
            summary_type=aai.SummarizationType.bullets,
            **speaker_separation_param,  # type: ignore[arg-type]
        ),
    )

    # The SDK must not raise before submitting
    assert request_body.get("summarization") is True
    assert request_body.get("summary_model") == aai.SummarizationModel.conversational
    assert request_body.get("summary_type") == aai.SummarizationType.bullets

    # The speaker-separation flag must be forwarded to the API
    for key, value in speaker_separation_param.items():
        assert request_body.get(key) == value

    # Response must be parsed correctly
    assert transcript.error is None
    assert transcript.summary == mock_response["summary"]
