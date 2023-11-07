import factory
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class SentimentFactory(factories.WordFactory):
    sentiment = factory.Faker("enum", enum_cls=aai.types.SentimentType)
    speaker = factory.Faker("name")


class SentimentAnalysisResponseFactory(factories.TranscriptCompletedResponseFactory):
    sentiment_analysis_results = factory.List([factory.SubFactory(SentimentFactory)])


def test_sentiment_analysis_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `sentiment_analysis` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("sentiment_analysis") is None
    assert transcript.sentiment_analysis is None


def test_sentiment_analysis_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `sentiment_analysis=True` in the `TranscriptionConfig`
    will result in `sentiment_analysis=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(SentimentAnalysisResponseFactory)()
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(sentiment_analysis=True),
    )

    # Check that request body was properly defined
    assert request_body.get("sentiment_analysis") is True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None

    assert transcript.sentiment_analysis is not None
    assert len(transcript.sentiment_analysis) > 0
    assert len(transcript.sentiment_analysis) == len(
        mock_response["sentiment_analysis_results"]
    )

    for response_sentiment_result, transcript_sentiment_result in zip(
        mock_response["sentiment_analysis_results"],
        transcript.sentiment_analysis,
    ):
        assert transcript_sentiment_result.text == response_sentiment_result["text"]
        assert transcript_sentiment_result.start == response_sentiment_result["start"]
        assert transcript_sentiment_result.end == response_sentiment_result["end"]
        assert (
            transcript_sentiment_result.confidence
            == response_sentiment_result["confidence"]
        )
        assert (
            transcript_sentiment_result.sentiment.value
            == response_sentiment_result["sentiment"]
        )
        assert (
            transcript_sentiment_result.speaker == response_sentiment_result["speaker"]
        )
