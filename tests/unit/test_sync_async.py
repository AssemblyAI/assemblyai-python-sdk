import asyncio

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

TRANSCRIBE_URL = f"{aai.settings.sync_base_url}/v1/transcribe"
WARM_URL = f"{aai.settings.sync_base_url}/v1/warm"

_OK_RESPONSE = {
    "text": "hello world",
    "words": [
        {"text": "hello", "start": 0, "end": 200, "confidence": 0.9},
        {"text": "world", "start": 220, "end": 400, "confidence": 0.95},
    ],
    "confidence": 0.92,
    "audio_duration_ms": 400,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 243.7,
}


def _mock_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_OK_RESPONSE,
    )


async def test_transcribe_bytes_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing raw audio bytes
    async with aai.AsyncSyncTranscriber() as transcriber:
        result = await transcriber.transcribe(b"RIFFfake-wav-bytes")

    # Then the response is parsed into a SyncTranscriptResponse
    assert isinstance(result, aai.SyncTranscriptResponse)
    assert result.text == "hello world"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[0].start == 0
    assert result.words[0].end == 200
    assert result.words[1].text == "world"
    assert result.request_time_ms == 243.7


async def test_transcribe_sends_model_header_and_wav_part(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with the default config
    async with aai.AsyncSyncTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes")

    # Then the request routes via X-AAI-Model and ships a WAV audio part
    request = httpx_mock.get_requests()[0]
    assert request.headers["X-AAI-Model"] == "universal-3-5-pro"
    body = request.read()
    assert b'name="audio"' in body
    assert b"Content-Type: audio/wav" in body
    # And no config part is sent when the config is empty
    assert b'name="config"' not in body


async def test_transcribe_sends_config_part(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a prompt and keyterms_prompt
    config = aai.SyncTranscriptionConfig(
        prompt="Transcribe verbatim.",
        keyterms_prompt=["AssemblyAI"],
    )
    async with aai.AsyncSyncTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then a config JSON part carries the options
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b"Transcribe verbatim." in body
    assert b'"AssemblyAI"' in body
    # And the routing model is never placed in the body
    assert b'"model"' not in body


async def test_transcribe_uses_default_config_and_per_call_override(
    httpx_mock: HTTPXMock,
):
    # Given a transcriber with a default config
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    default = aai.SyncTranscriptionConfig(prompt="default prompt")

    async with aai.AsyncSyncTranscriber(config=default) as transcriber:
        # When transcribing without a per-call config
        await transcriber.transcribe(b"RIFFfake-wav-bytes")
        # And with a per-call override
        override = aai.SyncTranscriptionConfig(prompt="override prompt")
        await transcriber.transcribe(b"RIFFfake-wav-bytes", config=override)

    # Then the default applies to the first call and the override to the second
    first, second = (request.read() for request in httpx_mock.get_requests())
    assert b"default prompt" in first
    assert b"override prompt" in second


async def test_transcribe_pcm_sends_pcm_part_and_rate(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with sample_rate + channels (raw PCM)
    config = aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
    async with aai.AsyncSyncTranscriber() as transcriber:
        await transcriber.transcribe(b"\x00\x01" * 100, config=config)

    # Then the audio part is PCM and the config carries rate + channels
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


async def test_transcribe_pcm_without_rate_raises():
    # Given a config with sample_rate but no channels (partial PCM intent)
    config = aai.SyncTranscriptionConfig(sample_rate=16000)

    # When transcribing, Then it fails locally before any request
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(ValueError, match="sample_rate and channels"):
            await transcriber.transcribe(b"\x00\x01" * 100, config=config)


async def test_transcribe_rejects_url():
    # Given an http URL as input
    async with aai.AsyncSyncTranscriber() as transcriber:
        # When transcribing, Then it is rejected with a pointer to Transcriber
        with pytest.raises(ValueError, match="does not accept URLs"):
            await transcriber.transcribe("https://example.com/audio.wav")


async def test_transcribe_path_input(httpx_mock: HTTPXMock, tmp_path):
    # Given a local WAV file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"RIFFfake-wav-bytes")

    # When transcribing the path
    async with aai.AsyncSyncTranscriber() as transcriber:
        result = await transcriber.transcribe(str(audio_file))

    # Then it succeeds and ships the file under its own name
    assert result.text == "hello world"
    body = httpx_mock.get_requests()[0].read()
    assert b'filename="call.wav"' in body


async def test_transcribe_gather_runs_concurrently(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint answering twice
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)

    # When fanning two clips out with asyncio.gather on one transcriber
    async with aai.AsyncSyncTranscriber() as transcriber:
        results = await asyncio.gather(
            transcriber.transcribe(b"RIFFone"),
            transcriber.transcribe(b"RIFFtwo"),
        )

    # Then both finish and parse
    assert [result.text for result in results] == ["hello world", "hello world"]


async def test_problem_details_envelope_maps_to_sync_transcript_error(
    httpx_mock: HTTPXMock,
):
    # Given the server rejects oversized audio with a problem-details body
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=413,
        json={"status": 413, "title": "Audio Too Large", "detail": "too long"},
    )

    # When transcribing, Then a SyncTranscriptError carries the snake_cased
    # title as error_code, plus the status and detail
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(aai.SyncTranscriptError) as exc_info:
            await transcriber.transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


async def test_rate_limit_surfaces_retry_after(httpx_mock: HTTPXMock):
    # Given a rate-limit response with a Retry-After header
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=429,
        json={
            "status": 429,
            "title": "Too Many Requests",
            "detail": "Too many requests",
        },
        headers={"Retry-After": "5"},
    )

    # When transcribing, Then retry_after and the snake_cased title are parsed
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(aai.SyncTranscriptError) as exc_info:
            await transcriber.transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


async def test_warm_opens_connection_with_model_header(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    async with aai.AsyncSyncTranscriber() as transcriber:
        warmed = await transcriber.warm()

    # Then it returns True and routes the probe via X-AAI-Model
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert request.url == WARM_URL
    assert request.method == "GET"
    assert request.headers["X-AAI-Model"] == "universal-3-5-pro"


async def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    # Given a warm route that the load balancer answers with a 404
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    # When warming, Then the socket is still established, so warm() is True
    async with aai.AsyncSyncTranscriber() as transcriber:
        assert await transcriber.warm() is True


async def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    # Given the sync host is unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    # When warming, Then the failure is swallowed and reported as False
    async with aai.AsyncSyncTranscriber() as transcriber:
        assert await transcriber.warm() is False


async def test_context_manager_closes_owned_client():
    # Given a transcriber that created its own client
    async with aai.AsyncSyncTranscriber() as transcriber:
        assert isinstance(transcriber, aai.AsyncSyncTranscriber)
        assert not transcriber.client.http_client.is_closed

    # Then leaving the block closes the owned connection pool
    assert transcriber.client.http_client.is_closed


async def test_aclose_leaves_shared_client_open():
    # Given a transcriber built on a caller-owned client
    async with aai.AsyncClient(settings=aai.settings) as client:
        transcriber = aai.AsyncSyncTranscriber(client=client)

        # When closing the transcriber
        await transcriber.aclose()

        # Then the shared pool stays open — its creator closes it
        assert not client.http_client.is_closed

    assert client.http_client.is_closed
