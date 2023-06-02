from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx
import pytest
from faker import Faker
from pytest_httpx import HTTPXMock

import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


def test_export_subtitles_succeeds(httpx_mock: HTTPXMock, faker: Faker):
    """
    Tests whether exporting subtitles succeed.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    expected_subtitles_srt = faker.text()
    expected_subtitles_vtt = faker.text()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/srt",
        status_code=httpx.codes.OK,
        method="GET",
        text=expected_subtitles_srt,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/vtt",
        status_code=httpx.codes.OK,
        method="GET",
        text=expected_subtitles_vtt,
    )

    srt_subtitles = transcript.export_subtitles_srt()
    vtt_subtitles = transcript.export_subtitles_vtt()

    assert srt_subtitles == expected_subtitles_srt
    assert vtt_subtitles == expected_subtitles_vtt

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 2


def test_export_subtitles_fails(httpx_mock: HTTPXMock):
    """
    Tests whether exporting subtitles fails.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/srt",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json={"error": "something went wrong"},
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/vtt",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json={"error": "something went wrong"},
    )

    with pytest.raises(aai.TranscriptError, match="something went wrong"):
        transcript.export_subtitles_srt()

    with pytest.raises(aai.TranscriptError, match="something went wrong"):
        transcript.export_subtitles_vtt()

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 2


def test_word_search_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether word search succeeds.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # create a mock response for the word search
    mock_word_search_response = factories.generate_dict_factory(
        factories.WordSearchMatchResponseFactory
    )()

    search_words = {
        "words": ",".join(["test", "me"]),
    }
    # mock the specific endpoints
    url = httpx.URL(
        f"{aai.settings.base_url}/transcript/{transcript.id}/word-search?{urlencode(search_words)}",
    )

    httpx_mock.add_response(
        url=url,
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_word_search_response,
    )

    # mimic the SDK call
    matches = transcript.word_search(words=["test", "me"])

    # check integrity of the response

    for idx, word_search in enumerate(matches):
        assert isinstance(word_search, aai.WordSearchMatch)
        assert word_search.count == mock_word_search_response["matches"][idx]["count"]
        assert (
            word_search.timestamps
            == mock_word_search_response["matches"][idx]["timestamps"]
        )
        assert word_search.text == mock_word_search_response["matches"][idx]["text"]
        assert (
            word_search.indexes == mock_word_search_response["matches"][idx]["indexes"]
        )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_word_search_fails(httpx_mock: HTTPXMock):
    """
    Tests whether word search fails.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # mock the specific endpoints
    url = httpx.URL(
        f"{aai.settings.base_url}/transcript/{transcript.id}/word-search?words=test",
    )

    httpx_mock.add_response(
        url=url,
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json={"error": "something went wrong"},
    )

    with pytest.raises(aai.TranscriptError, match="something went wrong"):
        transcript.word_search(words=["test"])

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_get_sentences_and_paragraphs_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether getting sentences and paragraphs succeeds.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # create a mock response for the sentences
    mock_sentences_response = factories.generate_dict_factory(
        factories.SentencesResponseFactory
    )()

    # create a mock response for the paragraphs
    mock_paragraphs_response = factories.generate_dict_factory(
        factories.ParagraphsResponseFactory
    )()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/sentences",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_sentences_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/paragraphs",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_paragraphs_response,
    )

    # mimic the SDK call
    sentences = transcript.get_sentences()
    paragraphs = transcript.get_paragraphs()

    # check integrity of the response
    def compare_words(lhs: List[aai.Word], rhs: List[Dict[str, Any]]) -> bool:
        """
        Compares the list of Word objects with the list of dicts.

        Args:
            lhs: The list of Word objects.
            rhs: The list of dicts.

        Returns:
            True if the lists are equal, False otherwise.
        """
        for idx, word in enumerate(lhs):
            if word.text != rhs[idx]["text"]:
                return False
            if word.start != rhs[idx]["start"]:
                return False
            if word.end != rhs[idx]["end"]:
                return False
        return True

    for idx, sentence in enumerate(sentences):
        assert isinstance(sentence, aai.Sentence)
        assert compare_words(
            sentence.words, mock_sentences_response["sentences"][idx]["words"]
        )

    for idx, paragraph in enumerate(paragraphs):
        assert isinstance(paragraph, aai.Paragraph)
        assert compare_words(
            paragraph.words, mock_paragraphs_response["paragraphs"][idx]["words"]
        )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 2


def test_get_sentences_and_paragraphs_fails(httpx_mock: HTTPXMock):
    """
    Tests whether getting sentences and paragraphs fails.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    transcript = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_transcript_response),
    )

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/sentences",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json={"error": "something went wrong"},
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{transcript.id}/paragraphs",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json={"error": "something went wrong"},
    )

    # mimic the SDK call
    with pytest.raises(aai.TranscriptError, match="something went wrong"):
        transcript.get_sentences()
    with pytest.raises(aai.TranscriptError, match="something went wrong"):
        transcript.get_paragraphs()

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 2
