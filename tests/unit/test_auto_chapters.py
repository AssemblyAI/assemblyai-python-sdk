import json
from typing import Any, Dict, Tuple

import factory
import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class AutoChaptersResponseFactory(factories.TranscriptCompletedResponseFactory):
    chapters = factory.List([factory.SubFactory(factories.ChapterFactory)])


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


def test_auto_chapters_fails_without_punctuation(httpx_mock: HTTPXMock):
    """
    Tests whether the SDK raises an error before making a request
    if `auto_chapters` is enabled and `punctuation` is disabled
    """

    with pytest.raises(ValueError) as error:
        __submit_mock_request(
            httpx_mock,
            mock_response={},  # response doesn't matter, since it shouldn't occur
            config=aai.TranscriptionConfig(
                auto_chapters=True,
                punctuate=False,
            ),
        )
    # Check that the error message informs the user of the invalid parameter
    assert "punctuate" in str(error)

    # Check that the error was raised before any requests were made
    assert len(httpx_mock.get_requests()) == 0

    # Inform httpx_mock that it's okay we didn't make any requests
    httpx_mock.reset(False)


def test_auto_chapters_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `auto_chapters` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("auto_chapters") is None
    assert transcript.chapters is None


def test_auto_chapters_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `auto_chapters=True` in the `TranscriptionConfig`
    will result in `auto_chapters=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(AutoChaptersResponseFactory)()
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(auto_chapters=True),
    )

    # Check that request body was properly defined
    assert request_body.get("auto_chapters") == True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.chapters is not None
    assert len(transcript.chapters) > 0
    assert len(transcript.chapters) == len(mock_response["chapters"])

    for response_chapter, transcript_chapter in zip(
        mock_response["chapters"], transcript.chapters
    ):
        assert transcript_chapter.summary == response_chapter["summary"]
        assert transcript_chapter.headline == response_chapter["headline"]
        assert transcript_chapter.gist == response_chapter["gist"]
        assert transcript_chapter.start == response_chapter["start"]
        assert transcript_chapter.end == response_chapter["end"]
