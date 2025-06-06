import copy
import json
import os
from unittest.mock import mock_open, patch
from urllib.parse import urlencode

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


def test_upload_file_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the submission of a file fails.
    """

    # our local audio file that we want to transcribe
    local_file = os.urandom(10)

    # this is the url that we should receive when uploading the local file
    expected_upload_url = "https://example.org/audio.wav"

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
        status_code=httpx.codes.OK,
        method="POST",
        json={"upload_url": expected_upload_url},
        match_content=local_file,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()

    # patch the reading of a local file
    with patch("builtins.open", mock_open(read_data=local_file)):
        audio_url = transcriber.upload_file("audio.wav")

    # check whether returned audio_url is correct
    assert audio_url == expected_upload_url


def test_upload_file_fails(httpx_mock: HTTPXMock):
    """
    Tests whether the submission of a file fails.
    """

    returned_error_message = "something went wrong"

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": returned_error_message},
    )

    # check whether uploading a file raises a TranscriptError
    with pytest.raises(aai.TranscriptError) as excinfo:
        aai.Transcriber().upload_file(os.urandom(10))

    # check wheter the TranscriptError contains the specified error message
    assert returned_error_message in str(excinfo.value)
    assert httpx.codes.INTERNAL_SERVER_ERROR == excinfo.value.status_code


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
    with pytest.raises(aai.TranscriptError) as excinfo:
        transcriber.submit("https://example.org/audio.wav")

    assert "something went wrong" in str(excinfo)
    assert httpx.codes.INTERNAL_SERVER_ERROR == excinfo.value.status_code

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
        with pytest.raises(aai.TranscriptError) as excinfo:
            transcriber.transcribe("audio.wav")

    # check wheter the Exception contains the specified error message
    assert "something went wrong" in str(excinfo.value)
    assert httpx.codes.INTERNAL_SERVER_ERROR == excinfo.value.status_code

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


def test_transcribe_file_binary_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether the transcription of a binary file works.
    """

    # create a mock response of a completed transcript
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    # our binary audio file that we want to transcribe
    file_data = os.urandom(10)

    # this is the url that we should receive when uploading the local file
    expected_upload_url = "https://example.org/audio.wav"
    mock_transcript_response["audio_url"] = expected_upload_url

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
        status_code=httpx.codes.OK,
        method="POST",
        json={"upload_url": expected_upload_url},
        match_content=file_data,
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

    transcript = transcriber.transcribe(file_data)

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
    fails_json = {"error": "something went wrong"}

    # mock the specific endpoints

    # the first file fails (see `.transcripe_group(...)` below)
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json=fails_json,
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
    transcript_group, failures = transcriber.transcribe_group(
        [
            expect_failed_audio_url,
            expect_succeeded_audio_url,
        ],
        return_failures=True,
    )

    # ensure integrity
    assert len(transcript_group.transcripts) == 1
    assert len(failures) == 1

    # Check whether the error message corresponds to the raised TranscriptError message
    assert "failed to transcribe url" in str(failures[0])
    assert failures[0].status_code == httpx.codes.INTERNAL_SERVER_ERROR


def test_transcribe_group_urls_fails_during_polling(httpx_mock: HTTPXMock):
    """
    Tests the scenario of when a list of paths or URLs are being transcribed
    and one (or more) items of that list fail during the polling process.

    In this case we need to ensure that the application flow does not interrupt unexpectedly and the user
    is able to backtrace the reason
    """
    # create a mock response for two processing transcripts
    mock_processing_response_1 = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    mock_processing_response_2 = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    expected_audio_urls = ["https://example.org/1.wav", "https://example.org/2.wav"]
    mock_processing_response_1["audio_url"] = expected_audio_urls[0]
    mock_processing_response_2["audio_url"] = expected_audio_urls[1]

    # mock both URLs succeeding on submission
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_processing_response_1,
    )
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_processing_response_2,
    )

    # create a mock response for a successful GET and a failed GET
    mock_succeeds_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    failed_json = {"error": "something went wrong"}

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_processing_response_1['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_succeeds_response,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_processing_response_2['id']}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="GET",
        json=failed_json,
    )

    # mimic the usage of the SDK
    transcriber = aai.Transcriber()
    transcript_group, failures = transcriber.transcribe_group(
        expected_audio_urls,
        return_failures=True,
    )

    # ensure integrity
    assert len(transcript_group.transcripts) == 1
    assert len(failures) == 1

    # Check whether the error message is correct
    assert "failed to retrieve transcript" in str(failures[0])
    assert failures[0].status_code == httpx.codes.INTERNAL_SERVER_ERROR


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
    mock_completed_json = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_completed_json,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_completed_json['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_completed_json,
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


def test_language_code_string(httpx_mock: HTTPXMock):
    mock_completed_json = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_completed_json,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_completed_json['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_completed_json,
    )

    aai.Transcriber().transcribe(
        "https://example.org/audio.wav",
        config=aai.TranscriptionConfig(
            language_code="en",
        ),
    )

    request = json.loads(httpx_mock.get_requests()[0].content.decode())
    assert request.get("language_code") == "en"


def test_language_code_enum(httpx_mock: HTTPXMock):
    mock_completed_json = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_completed_json,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_completed_json['id']}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_completed_json,
    )

    with pytest.deprecated_call():
        language_code = aai.LanguageCode.en

    transcript = aai.Transcriber().transcribe(
        "https://example.org/audio.wav",
        config=aai.TranscriptionConfig(
            language_code=language_code,
        ),
    )

    assert transcript.config.language_code == language_code


def test_list_transcripts(httpx_mock: HTTPXMock):
    mock_list_transcript_response = factories.generate_dict_factory(
        factories.ListTranscriptResponse
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_list_transcript_response,
    )

    page = aai.Transcriber().list_transcripts()

    assert isinstance(page, aai.ListTranscriptResponse)
    assert page.page_details is not None
    assert page.transcripts is not None

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_list_transcripts_parameters(httpx_mock: HTTPXMock):
    mock_list_transcript_response = factories.generate_dict_factory(
        factories.ListTranscriptResponse
    )()

    params = aai.ListTranscriptParameters(
        limit=2,
        status=aai.TranscriptStatus.completed,
    )

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}?{urlencode(params.dict(exclude_none=True))}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_list_transcript_response,
    )

    page = aai.Transcriber().list_transcripts(params)

    assert isinstance(page, aai.ListTranscriptResponse)
    assert page.page_details is not None
    assert page.transcripts is not None

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1
