import factory
import pytest
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class AutoChaptersResponseFactory(factories.TranscriptCompletedResponseFactory):
    chapters = factory.List([factory.SubFactory(factories.ChapterFactory)])


def test_auto_chapters_fails_without_punctuation(httpx_mock: HTTPXMock):
    """
    Tests whether the SDK raises an error before making a request
    if `auto_chapters` is enabled and `punctuation` is disabled
    """

    with pytest.raises(ValueError) as error:
        unit_test_utils.submit_mock_transcription_request(
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
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
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
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(auto_chapters=True),
    )

    # Check that request body was properly defined
    assert request_body.get("auto_chapters") is True

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
