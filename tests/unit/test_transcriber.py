import copy
import json
import os
from unittest.mock import mock_open, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import (
    ENDPOINT_TRANSCRIPT,
    ENDPOINT_UPLOAD,
)
from tests.unit import factories

aai.settings.api_key = "test"


def test_submit_url_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the submission of an URL works.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_transcript_response,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript = transcriber.submit("https://example.org/audio.wav")

    # ensure integrity
    assert transcript.id == mock_transcript_response["id"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_submit_url_fails(httpx_mock: HTTPXMock):
    """
    Tests whether the submission of an URL fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript = transcriber.submit("https://example.org/audio.wav")

    # check whether the status is set to error
    assert transcript.audio_url == "https://example.org/audio.wav"
    assert transcript.status == aai.TranscriptStatus.error
    assert "something went wrong" in transcript.error

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_submit_file_fails_due_api_error(httpx_mock: HTTPXMock):
    """
    Tests whether the submission of a file fails due to an API error.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()

    # patch the reading of a local file
    with patch("builtins.open", mock_open(read_data=os.urandom(10))):
        transcript = transcriber.transcribe("audio.wav")

    # check whether the status is set to error
    assert transcript.audio_url == "audio.wav"
    assert transcript.status == aai.TranscriptStatus.error
    assert "something went wrong" in transcript.error

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_submit_file_fails_due_file_not_found():
    """
    Tests whether the submission of a file fails due to a file not found error.
    """

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()

    # check whether the file not found error is raised
    with pytest.raises(FileNotFoundError):
        transcriber.submit("audio.wav")


def test_transcribe_url_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the transcription of an URL works.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_transcript_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_transcript_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_transcript_response,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe("https://example.org/audio.wav")

    # ensure integrity
    assert transcript.id == mock_transcript_response["id"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 2


def test_transcribe_file_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the transcription of a local file works.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # our local audio file that we want to transcribe
    local_file = os.urandom(10)

    # this is the url that we should receive when uploading the local file
    expected_upload_url = "https://example.org/audio.wav"
    mock_transcript_response["audio_url"] = expected_upload_url

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
        status_code=httpx.codes.OK,
        method="POST",
        json={"upload_url": expected_upload_url},
        match_content=local_file,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_transcript_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_transcript_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_transcript_response,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()

    # patch the reading of a local file
    with patch("builtins.open", mock_open(read_data=local_file)):
        transcript = transcriber.transcribe("audio.wav")

    # ensure integrity
    assert transcript.id == mock_transcript_response["id"]
    assert transcript.audio_url == expected_upload_url

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 3


def test_transcribe_group_urls_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the transcription of multiple URLs work.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # create different response mock objects
    response_1 = copy.deepcopy(mock_transcript_response)
    response_2 = copy.deepcopy(mock_transcript_response)

    expected_audio_urls = ["https://example.org/1.wav", "https://example.org/2.wav"]
    response_1["audio_url"] = expected_audio_urls[0]
    response_2["audio_url"] = expected_audio_urls[1]

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=response_1,
    )
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=response_2,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{response_1['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=response_1,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{response_2['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=response_2,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript_group = transcriber.transcribe_group(expected_audio_urls)

    # ensure integrity
    assert len(transcript_group.transcripts) == 2

    # check whether the audio urls match
    assert {t.audio_url for t in transcript_group} == set(expected_audio_urls)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 4


def test_transcribe_group_urls_fails_during_upload(httpx_mock: HTTPXMock):
    """
    Tests the scenario of when a list of paths or URLs are being transcribed
    and one (or more) items of that list fail due a HTTP-error (e.g. Timeout, Internal Server Error, etc.)

    In this case we need to ensure that the application flow does not interrupt unexpectedly and the user
    is able to backtrace the reason of those failed uploads.
    """

    # create a mock response of a completed transcript
    succeeds_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    expect_succeeded_audio_url = "https://example.org/succeeds.wav"
    succeeds_response["audio_url"] = expect_succeeded_audio_url

    expect_failed_audio_url = "https://example.org/fails.wav"

    # mock the specific endpoints

    # the first file fails (see `.transcripe_group(...)` below)
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "Aww, Snap!"},
    )

    # the second one succeeds (see `.transcribe_group(...) below`)
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=succeeds_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{succeeds_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=succeeds_response,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript_group = transcriber.transcribe_group(
        [
            expect_failed_audio_url,
            expect_succeeded_audio_url,
        ],
    )

    # ensure integrity
    assert len(transcript_group.transcripts) == 2

    # check whether the audio urls match
    audio_urls = {t.audio_url for t in transcript_group.transcripts}
    assert audio_urls == {expect_failed_audio_url, expect_succeeded_audio_url}

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 4


def test_transcribe_async_url_succeeds(httpx_mock: HTTPXMock):
    # create a mock response of a processing transcript
    mock_processing_response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    # create a mock response of a completed transcript
    mock_completed_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    mock_completed_response["id"] = mock_processing_response["id"]

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_processing_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_processing_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_processing_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_processing_response['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_completed_response,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript_future = transcriber.transcribe_async(
        "https://example.org/audio.wav",
    )

    # `transcribe` is being called with a callback the operation works asynchronously
    transcript = transcript_future.result()

    # ensure integrity
    assert transcript.id == mock_completed_response["id"]
    assert transcript.status == aai.TranscriptStatus.completed

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 3


def test_transcribe_async_url_fails(httpx_mock: HTTPXMock):
    # create a mock response of a processing transcript
    mock_processing_transcript = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    # create a mock response of a error transcript
    mock_error_transcript = factories.generate_dict_factory(
        factories.TranscriptErrorResponseFactory
    )()

    mock_error_transcript["id"] = mock_processing_transcript["id"]

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_processing_transcript,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_processing_transcript['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_processing_transcript,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_error_transcript['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_error_transcript,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript_future = transcriber.transcribe_async(
        "https://example.org/audio.wav",
    )

    # `transcribe` is being called with a callback the operation works asynchronously
    transcript = transcript_future.result()

    # ensure integrity
    assert transcript.id == mock_error_transcript["id"]
    assert transcript.status == aai.TranscriptStatus.error

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 3


def test_language_detection(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json={},
    )

    aai.Transcriber().transcribe(
        "https://example.org/audio.wav",
        config=aai.TranscriptionConfig(
            language_code=None,
            language_detection=True,
        ),
    )

    request = json.loads(httpx_mock.get_requests()[0].content.decode())
    assert request["language_detection"] is True
    assert request.get("language_code") is None
