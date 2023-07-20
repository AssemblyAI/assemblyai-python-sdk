import json
from typing import Any, Dict, Tuple

import factory
import httpx
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories

aai.settings.api_key = "test"


class AutohighlightResultFactory(factory.Factory):
    class Meta:
        model = aai.types.AutohighlightResult

    count = factory.Faker("pyint")
    rank = factory.Faker("pyfloat")
    text = factory.Faker("sentence")
    timestamps = factory.List([factory.SubFactory(factories.TimestampFactory)])


class AutohighlightResponseFactory(factory.Factory):
    class Meta:
        model = aai.types.AutohighlightResponse

    status = aai.types.StatusResult.success
    results = factory.List([factory.SubFactory(AutohighlightResultFactory)])


class AutohighlightTranscriptResponseFactory(
    factories.TranscriptCompletedResponseFactory
):
    auto_highlights_result = factory.SubFactory(AutohighlightResponseFactory)


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


def test_auto_highlights_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `auto_highlights` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("auto_highlights") is None
    assert transcript.auto_highlights is None


def test_auto_highlights_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `auto_highlights=True` in the `TranscriptionConfig`
    will result in `auto_highlights=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(
        AutohighlightTranscriptResponseFactory
    )()
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(auto_highlights=True),
    )

    # Check that request body was properly defined
    assert request_body.get("auto_highlights") == True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.auto_highlights is not None
    assert (
        transcript.auto_highlights.status
        == mock_response["auto_highlights_result"]["status"]
    )

    assert transcript.auto_highlights.results is not None
    assert len(transcript.auto_highlights.results) > 0
    assert len(transcript.auto_highlights.results) == len(
        mock_response["auto_highlights_result"]["results"]
    )

    for response_result, transcript_result in zip(
        mock_response["auto_highlights_result"]["results"],
        transcript.auto_highlights.results,
    ):
        assert transcript_result.count == response_result["count"]
        assert transcript_result.rank == response_result["rank"]
        assert transcript_result.text == response_result["text"]

        for response_timestamp, transcript_timestamp in zip(
            response_result["timestamps"], transcript_result.timestamps
        ):
            assert transcript_timestamp.start == response_timestamp["start"]
            assert transcript_timestamp.end == response_timestamp["end"]
