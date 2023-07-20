import json
from typing import Any, Dict, Tuple

import factory
import httpx
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories

aai.settings.api_key = "test"


class SentimentFactory(factories.WordFactory):
    sentiment = factory.Faker("enum", enum_cls=aai.types.SentimentType)
    speaker = factory.Faker("name")


class SentimentAnalysisResponseFactory(factories.TranscriptCompletedResponseFactory):
    sentiment_analysis_results = factory.List([factory.SubFactory(SentimentFactory)])


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


def test_sentiment_analysis_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `sentiment_analysis` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = __submit_mock_request(
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
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(sentiment_analysis=True),
    )

    # Check that request body was properly defined
    assert request_body.get("sentiment_analysis") == True

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
