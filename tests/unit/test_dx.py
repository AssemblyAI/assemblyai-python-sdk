"""Tests for the client developer-experience surface.

Covers the pieces shared across the four transcribers and the two streaming
clients rather than any single product: `api_key=` construction, the
config-type guard, the audio input types the sync `Transcriber` accepts,
`poll_timeout`, the sync API's error parsing, and the lazily imported
`aai.streaming` attribute.
"""

import io
import os
import subprocess
import sys

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT, ENDPOINT_UPLOAD
from assemblyai.streaming.v3 import (
    AsyncStreamingClient,
    StreamingClient,
    StreamingClientOptions,
)
from tests.unit import factories

aai.settings.api_key = "test"

TRANSCRIPT_URL = f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}"
UPLOAD_URL = f"{aai.settings.base_url}{ENDPOINT_UPLOAD}"
SYNC_TRANSCRIBE_URL = f"{aai.settings.sync_base_url}/v1/transcribe"

_SYNC_OK_RESPONSE = {
    "text": "hello world",
    "words": [{"text": "hello", "start": 0, "end": 200, "confidence": 0.9}],
    "confidence": 0.92,
    "audio_duration_ms": 400,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 243.7,
}


@pytest.fixture
def no_global_api_key():
    """Clears the global API key, so only an explicit `api_key=` can authenticate."""

    original = aai.settings.api_key
    aai.settings.api_key = None
    yield
    aai.settings.api_key = original


@pytest.fixture
def fast_polling():
    """Keeps the polling loop from spending seconds sleeping in tests."""

    original = aai.settings.polling_interval
    aai.settings.polling_interval = 0.001
    yield
    aai.settings.polling_interval = original


def _completed_response(**overrides) -> dict:
    response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    response.update(overrides)

    return response


def _processing_response(transcript_id: str) -> dict:
    response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()
    response["id"] = transcript_id

    return response


def _mock_submit(httpx_mock: HTTPXMock, response: dict) -> None:
    httpx_mock.add_response(
        url=TRANSCRIPT_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )


def _mock_poll(httpx_mock: HTTPXMock, response: dict) -> None:
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{response['id']}",
        method="GET",
        status_code=httpx.codes.OK,
        json=response,
    )


# == api_key= ==


def test_transcriber_accepts_api_key(no_global_api_key):
    # When constructing a Transcriber with only an explicit key
    transcriber = aai.Transcriber(api_key="explicit-key")

    # Then the client authenticates with it and the global settings are untouched
    assert transcriber._client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_sync_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.SyncTranscriber(api_key="explicit-key")

    assert transcriber._client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


@pytest.mark.asyncio
async def test_async_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.AsyncTranscriber(api_key="explicit-key")
    try:
        # Then the client authenticates with it and is owned by the transcriber
        assert transcriber.client.settings.api_key == "explicit-key"
        assert transcriber._owns_client is True
        assert aai.settings.api_key is None
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed


@pytest.mark.asyncio
async def test_async_sync_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.AsyncSyncTranscriber(api_key="explicit-key")
    try:
        assert transcriber.client.settings.api_key == "explicit-key"
        assert transcriber._owns_client is True
        assert aai.settings.api_key is None
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed


def _caller_settings() -> aai.Settings:
    """Settings with a distinguishable field, to prove they survive a derive."""

    return aai.Settings(api_key="from-client", http_timeout=42.5)


@pytest.mark.parametrize("transcriber_class", [aai.Transcriber, aai.SyncTranscriber])
def test_api_key_takes_precedence_over_a_given_client(transcriber_class):
    caller_client = aai.Client(settings=_caller_settings())

    transcriber = transcriber_class(client=caller_client, api_key="explicit-key")

    # Then a derived client is used, not the caller's
    assert transcriber._client is not caller_client
    assert transcriber._client.settings.api_key == "explicit-key"
    # And the caller's client is untouched, with its other settings carried over
    assert caller_client.settings.api_key == "from-client"
    assert transcriber._client.settings.http_timeout == 42.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcriber_class", [aai.AsyncTranscriber, aai.AsyncSyncTranscriber]
)
async def test_async_api_key_takes_precedence_over_a_given_client(transcriber_class):
    caller_client = aai.AsyncClient(settings=_caller_settings())

    transcriber = transcriber_class(client=caller_client, api_key="explicit-key")
    try:
        assert transcriber.client is not caller_client
        assert transcriber.client.settings.api_key == "explicit-key"
        assert caller_client.settings.api_key == "from-client"
        assert transcriber.client.settings.http_timeout == 42.5
        # And the derived client is the transcriber's to close
        assert transcriber._owns_client is True
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed
    # The caller's client is left open for the caller to close
    assert not caller_client.http_client.is_closed

    await caller_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcriber_class", [aai.AsyncTranscriber, aai.AsyncSyncTranscriber]
)
async def test_async_client_passed_alone_stays_the_callers(transcriber_class):
    async with aai.AsyncClient(settings=aai.settings) as caller_client:
        transcriber = transcriber_class(client=caller_client)

        assert transcriber.client is caller_client
        assert transcriber._owns_client is False

        await transcriber.aclose()

        # aclose() leaves a client it does not own alone
        assert not caller_client.http_client.is_closed


def test_client_without_arguments_uses_the_global_settings():
    # When constructing a client with no arguments
    client = aai.Client()

    # Then it copies the global settings rather than sharing them
    assert client.settings.api_key == aai.settings.api_key
    assert client.settings is not aai.settings


def test_client_accepts_api_key(no_global_api_key):
    # When constructing a client with only an explicit key
    client = aai.Client(api_key="explicit-key")

    # Then it authenticates with it and the global settings are untouched
    assert client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_async_client_accepts_api_key(no_global_api_key):
    client = aai.AsyncClient(api_key="explicit-key")

    assert client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_api_key_overrides_the_key_on_given_settings():
    # Given a settings object of the caller's own
    settings = aai.Settings(api_key="from-settings")

    # When a client is built from it with an explicit key
    client = aai.Client(settings=settings, api_key="explicit-key")

    # Then the explicit key wins and the caller's settings are unchanged
    assert client.settings.api_key == "explicit-key"
    assert settings.api_key == "from-settings"


def test_missing_api_key_names_every_way_to_provide_one(no_global_api_key):
    with pytest.raises(ValueError) as exc_info:
        aai.Client()

    message = str(exc_info.value)
    assert "ASSEMBLYAI_API_KEY" in message
    assert "aai.settings.api_key" in message
    assert "api_key=" in message


# == config type guard ==


def test_sync_transcriber_rejects_a_job_api_config():
    with pytest.raises(TypeError) as exc_info:
        aai.SyncTranscriber(config=aai.TranscriptionConfig())

    assert str(exc_info.value) == (
        "SyncTranscriber expects SyncTranscriptionConfig, got TranscriptionConfig. "
        "Use aai.SyncTranscriptionConfig."
    )


def test_sync_transcriber_rejects_a_job_api_config_per_call():
    transcriber = aai.SyncTranscriber()

    with pytest.raises(TypeError) as exc_info:
        transcriber.transcribe(b"RIFFfake-wav-bytes", config=aai.TranscriptionConfig())

    assert "SyncTranscriber expects SyncTranscriptionConfig" in str(exc_info.value)


def test_transcriber_rejects_a_sync_api_config():
    with pytest.raises(TypeError) as exc_info:
        aai.Transcriber(config=aai.SyncTranscriptionConfig())

    assert str(exc_info.value) == (
        "Transcriber expects TranscriptionConfig, got SyncTranscriptionConfig. "
        "Use aai.TranscriptionConfig."
    )


@pytest.mark.asyncio
async def test_async_transcriber_rejects_a_sync_api_config():
    with pytest.raises(TypeError) as exc_info:
        aai.AsyncTranscriber(config=aai.SyncTranscriptionConfig())

    assert str(exc_info.value) == (
        "AsyncTranscriber expects TranscriptionConfig, got SyncTranscriptionConfig. "
        "Use aai.TranscriptionConfig."
    )


@pytest.mark.asyncio
async def test_async_transcriber_rejects_a_sync_api_config_per_call():
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(TypeError) as exc_info:
            await transcriber.transcribe(
                "https://example.org/audio.wav",
                config=aai.SyncTranscriptionConfig(),
            )

    assert "AsyncTranscriber expects TranscriptionConfig" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_sync_transcriber_rejects_a_job_api_config():
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(TypeError) as exc_info:
            await transcriber.transcribe(
                b"RIFFfake-wav-bytes",
                config=aai.TranscriptionConfig(),
            )

    assert "AsyncSyncTranscriber expects SyncTranscriptionConfig" in str(exc_info.value)


def test_the_matching_config_class_is_accepted(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    httpx_mock.add_response(
        url=SYNC_TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_SYNC_OK_RESPONSE,
    )

    # When the config matches the transcriber
    transcriber = aai.SyncTranscriber(config=aai.SyncTranscriptionConfig())
    result = transcriber.transcribe(
        b"RIFFfake-wav-bytes",
        config=aai.SyncTranscriptionConfig(timestamps=True),
    )

    # Then it transcribes as usual
    assert result.text == "hello world"


def test_a_config_subclass_is_accepted():
    class CustomConfig(aai.SyncTranscriptionConfig):
        pass

    transcriber = aai.SyncTranscriber(config=CustomConfig())

    assert isinstance(transcriber.config, CustomConfig)


# == audio input types ==


def _expect_upload(httpx_mock: HTTPXMock, audio: bytes) -> None:
    """
    Arms the upload endpoint, asserting the body through `match_content`.

    The body is matched while the request is being sent, which is the only
    point a streamed upload is readable: a path is streamed from a file object
    that closes when `upload_file` returns, and the oldest supported httpx
    consumes the request stream lazily. Reading the captured request afterwards
    would be reading a closed file.
    """

    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
        match_content=audio,
    )


def test_upload_file_accepts_a_pathlike(httpx_mock: HTTPXMock, tmp_path):
    # Given a local file addressed by a `pathlib.Path`
    audio = os.urandom(64)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(audio)

    _expect_upload(httpx_mock, audio)

    # When uploading it
    upload_url = aai.Transcriber().upload_file(audio_path)

    # Then the file's bytes are what got sent, and only that request was made
    assert upload_url == "https://example.org/uploaded.wav"
    assert len(httpx_mock.get_requests()) == 1


def test_upload_file_accepts_a_bytearray(httpx_mock: HTTPXMock):
    audio = bytearray(os.urandom(64))

    _expect_upload(httpx_mock, bytes(audio))

    aai.Transcriber().upload_file(audio)

    assert len(httpx_mock.get_requests()) == 1


def test_upload_file_accepts_a_file_object(httpx_mock: HTTPXMock):
    audio = os.urandom(64)

    _expect_upload(httpx_mock, audio)

    aai.Transcriber().upload_file(io.BytesIO(audio))

    assert len(httpx_mock.get_requests()) == 1


def test_upload_file_rejects_unsupported_input():
    with pytest.raises(TypeError) as exc_info:
        aai.Transcriber().upload_file(42)

    assert str(exc_info.value) == "unsupported audio input type: int"


def test_upload_file_still_raises_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        aai.Transcriber().upload_file(str(tmp_path / "absent.wav"))

    with pytest.raises(FileNotFoundError):
        aai.Transcriber().upload_file(tmp_path / "absent.wav")


# == poll_timeout ==


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_poll_timeout_raises_with_the_transcript_id(
    httpx_mock: HTTPXMock,
    fast_polling,
):
    # Given a job that never leaves `processing`
    _mock_submit(httpx_mock, _processing_response("stuck-id"))
    _mock_poll(httpx_mock, _processing_response("stuck-id"))

    # When transcribing with a short poll timeout
    with pytest.raises(aai.TranscriptError) as exc_info:
        aai.Transcriber().transcribe(
            "https://example.org/audio.wav",
            poll_timeout=0.05,
        )

    # Then the error names the transcript and its last-seen status
    message = str(exc_info.value)
    assert "stuck-id" in message
    assert "processing" in message


def test_poll_timeout_of_none_still_completes(httpx_mock: HTTPXMock, fast_polling):
    # Given a job that completes after a couple of polls
    completed = _completed_response()
    _mock_submit(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    # When transcribing without a poll timeout
    transcript = aai.Transcriber().transcribe(
        "https://example.org/audio.wav",
        poll_timeout=None,
    )

    assert transcript.status == aai.TranscriptStatus.completed


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_transcribe_async_accepts_poll_timeout(httpx_mock: HTTPXMock, fast_polling):
    # Given a job that never leaves `processing`
    _mock_submit(httpx_mock, _processing_response("stuck-id"))
    _mock_poll(httpx_mock, _processing_response("stuck-id"))

    future = aai.Transcriber().transcribe_async(
        "https://example.org/audio.wav",
        poll_timeout=0.05,
    )

    with pytest.raises(aai.TranscriptError) as exc_info:
        future.result()

    assert "stuck-id" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_async_poll_timeout_raises_with_the_transcript_id(
    httpx_mock: HTTPXMock,
    fast_polling,
):
    # Given a job that never leaves `processing`
    _mock_submit(httpx_mock, _processing_response("stuck-id"))
    _mock_poll(httpx_mock, _processing_response("stuck-id"))

    # When transcribing with a short poll timeout
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(aai.TranscriptError) as exc_info:
            await transcriber.transcribe(
                "https://example.org/audio.wav",
                poll_timeout=0.05,
            )

    message = str(exc_info.value)
    assert "stuck-id" in message
    assert "processing" in message


@pytest.mark.asyncio
async def test_async_poll_timeout_of_none_still_completes(
    httpx_mock: HTTPXMock,
    fast_polling,
):
    completed = _completed_response()
    _mock_submit(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.transcribe(
            "https://example.org/audio.wav",
            poll_timeout=None,
        )

    assert transcript.status == aai.TranscriptStatus.completed


# == sync API error parsing ==


def test_sync_error_falls_back_to_a_bare_error_key(httpx_mock: HTTPXMock):
    # Given a 4xx whose body is a plain `{"error": ...}`
    httpx_mock.add_response(
        url=SYNC_TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.BAD_REQUEST,
        json={"error": "some message"},
    )

    # When transcribing
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the message comes from `error`, without inventing an error code
    assert "some message" in str(exc_info.value)
    assert exc_info.value.error_code is None
    assert exc_info.value.status_code == httpx.codes.BAD_REQUEST


def test_sync_error_prefers_problem_details_over_the_error_key(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SYNC_TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.BAD_REQUEST,
        json={"title": "Bad Audio", "detail": "the detail", "error": "some message"},
    )

    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    assert "the detail" in str(exc_info.value)
    assert exc_info.value.error_code == "bad_audio"


# == streaming clients: api_key= ==


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_accepts_api_key(client_class):
    # When constructing with only an explicit key
    client = client_class(api_key="explicit-key")

    # Then the options carry it and every other option keeps its default
    defaults = StreamingClientOptions(api_key="explicit-key")
    assert client._options.api_key == "explicit-key"
    assert client._options.token is None
    assert client._options == defaults


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_overrides_the_key_on_options(client_class):
    options = StreamingClientOptions(api_key="from-options")

    client = client_class(options, api_key="explicit-key")

    # Then the constructor key wins
    assert client._options.api_key == "explicit-key"


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_override_carries_every_other_option(client_class):
    options = StreamingClientOptions(
        api_key="from-options",
        token="temporary-token",
        api_host="streaming.example.org",
        connect_timeout=2.5,
        max_connection_retries=7,
        connection_retry_delay=1.25,
        terminate_timeout=9.5,
    )

    client = client_class(options, api_key="explicit-key")

    # Then only api_key differs from what the caller set
    resolved = client._options
    assert resolved.api_key == "explicit-key"
    assert resolved.token == "temporary-token"
    assert resolved.api_host == "streaming.example.org"
    assert resolved.connect_timeout == 2.5
    assert resolved.max_connection_retries == 7
    assert resolved.connection_retry_delay == 1.25
    assert resolved.terminate_timeout == 9.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_override_does_not_mutate_the_callers_options(client_class):
    options = StreamingClientOptions(api_key="from-options", connect_timeout=2.5)

    client = client_class(options, api_key="explicit-key")

    # Then the caller's own object is untouched, and a copy was used
    assert options.api_key == "from-options"
    assert client._options is not options
    assert client._options.connect_timeout == 2.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_still_accepts_options_only(client_class):
    # Given the options a caller builds today, positionally
    options = StreamingClientOptions(api_key="from-options", connect_timeout=2.5)

    client = client_class(options)

    # Then the very object passed in is what the client uses
    assert client._options is options
    assert client._options.connect_timeout == 2.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_still_accepts_a_token_in_options(client_class):
    client = client_class(StreamingClientOptions(token="temporary-token"))

    assert client._options.token == "temporary-token"
    assert client._options.api_key is None


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_without_credentials_names_both_fixes(client_class):
    with pytest.raises(ValueError) as exc_info:
        client_class()

    message = str(exc_info.value)
    assert "api_key=" in message
    assert "RealTimeTranscriberOptions" in message
    assert "token=" in message


# == streaming renames: RealTime* canonical, Streaming* aliases ==

_RENAMED_PAIRS = [
    ("RealTimeTranscriber", "StreamingClient"),
    ("AsyncRealTimeTranscriber", "AsyncStreamingClient"),
    ("RealTimeTranscriberOptions", "StreamingClientOptions"),
    ("RealTimeParameters", "StreamingParameters"),
    ("RealTimeSessionParameters", "StreamingSessionParameters"),
    ("RealTimeEvents", "StreamingEvents"),
    ("RealTimeError", "StreamingError"),
    ("RealTimeErrorCodes", "StreamingErrorCodes"),
]


@pytest.mark.parametrize(("new_name", "old_name"), _RENAMED_PAIRS)
def test_old_streaming_name_is_an_alias_of_the_new_one(new_name, old_name):
    from assemblyai.streaming import v3

    # Then both names resolve, to the very same object
    assert getattr(v3, new_name) is getattr(v3, old_name)
    # And both are exported
    assert new_name in v3.__all__
    assert old_name in v3.__all__


def test_renamed_streaming_imports_are_warning_free():
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module = importlib.import_module("assemblyai.streaming.v3")

        for new_name, old_name in _RENAMED_PAIRS:
            getattr(module, new_name)
            getattr(module, old_name)


def test_real_time_transcriber_constructs_with_api_key():
    from assemblyai.streaming.v3 import AsyncRealTimeTranscriber, RealTimeTranscriber

    for client_class in (RealTimeTranscriber, AsyncRealTimeTranscriber):
        client = client_class(api_key="explicit-key")

        assert client._options.api_key == "explicit-key"


def test_real_time_transcriber_options_is_accepted_by_both_new_constructors():
    from assemblyai.streaming.v3 import (
        AsyncRealTimeTranscriber,
        RealTimeTranscriber,
        RealTimeTranscriberOptions,
    )

    options = RealTimeTranscriberOptions(api_key="from-options", connect_timeout=2.5)

    for client_class in (RealTimeTranscriber, AsyncRealTimeTranscriber):
        client = client_class(options)

        assert client._options is options
        assert client._options.connect_timeout == 2.5


def test_isinstance_holds_across_both_names():
    from assemblyai.streaming.v3 import (
        RealTimeTranscriber,
        RealTimeTranscriberOptions,
        StreamingClient,
        StreamingClientOptions,
    )

    client = RealTimeTranscriber(StreamingClientOptions(api_key="explicit-key"))

    assert isinstance(client, StreamingClient)
    assert isinstance(client, RealTimeTranscriber)
    assert isinstance(client._options, RealTimeTranscriberOptions)
    assert isinstance(client._options, StreamingClientOptions)


# == lazy streaming attribute ==


def test_streaming_attribute_imports_lazily():
    # Given a fresh interpreter, so no other test's imports can mask the result
    script = (
        "import sys, assemblyai; "
        "assert 'websockets' not in sys.modules; "
        "import assemblyai as a; "
        "a.streaming.v3.StreamingClient; "
        "assert 'websockets' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError) as exc_info:
        aai.not_a_real_name

    assert (
        str(exc_info.value) == "module 'assemblyai' has no attribute 'not_a_real_name'"
    )
