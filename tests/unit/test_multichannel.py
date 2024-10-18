from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class MultichannelResponseFactory(factories.TranscriptCompletedResponseFactory):
    multichannel = True
    audio_channels = 2


def test_multichannel_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that not setting `multichannel=True` in the `TranscriptionConfig`
    will result in the default behavior of it being excluded from the request body.
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("multichannel") is None
    assert transcript.json_response.get("multichannel") is None


def test_multichannel_enabled(httpx_mock: HTTPXMock):
    """
    Tests that not setting `multichannel=True` in the `TranscriptionConfig`
    will result in correct `multichannel` in the request body, and that the
    response is properly parsed into the `multichannel` and `utterances` field.
    """

    mock_response = factories.generate_dict_factory(MultichannelResponseFactory)()
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(multichannel=True),
    )

    # Check that request body was properly defined
    multichannel_response = request_body.get("multichannel")
    assert multichannel_response is not None

    # Check that transcript has no errors and multichannel response is correctly returned
    assert transcript.error is None
    assert transcript.json_response["multichannel"] == multichannel_response
    assert transcript.json_response["audio_channels"] > 1

    # Check that utterances are correctly parsed
    assert transcript.utterances is not None
    assert len(transcript.utterances) > 0
    for utterance in transcript.utterances:
        assert int(utterance.channel) > 0
