import asyncio

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

TRANSCRIBE_URL = f"{aai.settings.dictation_base_url}/v1/transcribe"
WARM_URL = f"{aai.settings.dictation_base_url}/v1/warm"

_OK_RESPONSE = {
    "text": "patient reports mild headache",
    "words": [
        {"text": "patient", "confidence": 0.9},
        {"text": "reports", "confidence": 0.95},
    ],
    "confidence": 0.92,
    "llm_response": None,
    "llm_error": None,
    "audio_duration_ms": 400,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 243.7,
    "sync_time_ms": 180.2,
}


def _mock_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_OK_RESPONSE,
    )


async def test_transcribe_bytes_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing raw audio bytes
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe(b"RIFFfake-wav-bytes")

    # Then the response is parsed into a DictationResponse
    assert isinstance(result, aai.DictationResponse)
    assert result.text == "patient reports mild headache"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[1].text == "reports"
    assert result.request_time_ms == 243.7
    assert result.sync_time_ms == 180.2


async def test_transcribe_posts_to_dictation_base_url(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes")

    # Then the request goes to the dictation host with the raw key and no
    # routing header
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == f"{aai.settings.dictation_base_url}/v1/transcribe"
    assert request.headers["authorization"] == "test"
    assert "X-AAI-Model" not in request.headers


async def test_transcribe_sends_config_part_before_audio_part(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a config
    config = aai.DictationConfig(language_codes=["en"])
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config part precedes the audio part in the multipart body —
    # the server reads the config before it starts consuming the audio stream
    body = httpx_mock.get_requests()[0].read()
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"Content-Type: application/json" in body


async def test_transcribe_omits_config_part_when_empty(httpx_mock: HTTPXMock):
    # Given a default config with nothing set
    _mock_ok(httpx_mock)

    # When transcribing, Then no config part is sent
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes")
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' not in body


async def test_transcribe_uses_default_config_and_per_call_override(
    httpx_mock: HTTPXMock,
):
    # Given a transcriber with a default config
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    default = aai.DictationConfig(llm_instruction="default instruction")

    async with aai.AsyncDictationTranscriber(config=default) as transcriber:
        # When transcribing without a per-call config
        await transcriber.transcribe(b"RIFFfake-wav-bytes")
        # And with a per-call override
        override = aai.DictationConfig(llm_instruction="override instruction")
        await transcriber.transcribe(b"RIFFfake-wav-bytes", config=override)

    # Then the default applies to the first call and the override to the second
    first, second = (request.read() for request in httpx_mock.get_requests())
    assert b"default instruction" in first
    assert b"override instruction" in second


async def test_transcribe_pcm_sends_pcm_part_and_rate(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with sample_rate + channels (raw PCM)
    config = aai.DictationConfig(sample_rate=16000, channels=1)
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe(b"\x00\x01" * 100, config=config)

    # Then the audio part is PCM and the config carries rate + channels
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


async def test_transcribe_pcm_without_channels_raises():
    # Given a config with sample_rate but no channels (partial PCM intent)
    config = aai.DictationConfig(sample_rate=16000)

    # When transcribing, Then it fails locally before any request
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(ValueError, match="sample_rate and channels"):
            await transcriber.transcribe(b"\x00\x01" * 100, config=config)


async def test_transcribe_rejects_url():
    # Given an http URL as input
    async with aai.AsyncDictationTranscriber() as transcriber:
        # When transcribing, Then it is rejected with a pointer to Transcriber,
        # and the message names this client rather than its threaded twin
        with pytest.raises(
            ValueError, match="AsyncDictationTranscriber does not accept URLs"
        ):
            await transcriber.transcribe("https://example.com/audio.wav")


async def test_transcribe_maps_mp3_extension_to_audio_mpeg(
    httpx_mock: HTTPXMock, tmp_path
):
    # Given a local MP3 file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When transcribing the path
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe(str(audio_file))

    # Then the audio part carries the MP3 Content-Type and the file's name
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/mpeg" in body
    assert b'filename="note.mp3"' in body


async def test_transcribe_wav_path_input(httpx_mock: HTTPXMock, tmp_path):
    # Given a local WAV file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.wav"
    audio_file.write_bytes(b"RIFFfake-wav-bytes")

    # When transcribing the path
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe(str(audio_file))

    # Then it succeeds and ships a WAV audio part
    assert result.text == "patient reports mild headache"
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/wav" in body


async def test_transcribe_gather_runs_concurrently(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint answering twice
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)

    # When fanning two clips out with asyncio.gather on one transcriber
    async with aai.AsyncDictationTranscriber() as transcriber:
        results = await asyncio.gather(
            transcriber.transcribe(b"RIFFone"),
            transcriber.transcribe(b"RIFFtwo"),
        )

    # Then both finish and parse
    assert [result.text for result in results] == [
        "patient reports mild headache",
        "patient reports mild headache",
    ]


async def test_final_text_prefers_llm_response(httpx_mock: HTTPXMock):
    # Given a response carrying an LLM rewrite
    response = dict(_OK_RESPONSE)
    response["llm_response"] = "S: Mild headache."
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When transcribing with an llm_instruction
    config = aai.DictationConfig(llm_instruction="Format this as a SOAP note.")
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe(b"RIFFfake", config=config)

    # Then final_text is the rewrite and the raw transcript is still there
    assert result.final_text == "S: Mild headache."
    assert result.text == "patient reports mild headache"


async def test_final_text_falls_back_to_text(httpx_mock: HTTPXMock):
    # Given a response whose LLM pass failed
    response = dict(_OK_RESPONSE)
    response["llm_error"] = "llm timed out"
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When transcribing
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe(b"RIFFfake")

    # Then final_text falls back to the raw transcript
    assert result.final_text == "patient reports mild headache"
    assert result.llm_error == "llm timed out"


async def test_error_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given a server rejecting the audio with an {error, error_code} body
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=415,
        json={"error": "unsupported media type", "error_code": "bad_audio"},
    )

    # When transcribing, Then a DictationError carries code + status
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 415
    assert error.error_code == "bad_audio"
    # And it is not the sync API's error class
    assert not isinstance(error, aai.SyncTranscriptError)


async def test_problem_details_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given the server rejects oversized audio with a problem-details body
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=413,
        json={"status": 413, "title": "Audio Too Large", "detail": "too long"},
    )

    # When transcribing, Then the snake_cased title becomes the error_code
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


async def test_detail_only_envelope_on_invalid_key(httpx_mock: HTTPXMock):
    # Given an invalid key, which the Dictation API answers 404
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=404,
        json={"detail": "Invalid API key"},
    )

    # When transcribing, Then the detail becomes the message
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 404
    assert error.error_code is None
    assert "Invalid API key" in str(error)


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
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


async def test_warm_opens_connection(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    async with aai.AsyncDictationTranscriber() as transcriber:
        warmed = await transcriber.warm()

    # Then it returns True and probes the dictation warm route
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == WARM_URL
    assert request.method == "GET"
    assert "X-AAI-Model" not in request.headers


async def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    # Given a warm route that the load balancer answers with a 404
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    # When warming, Then the socket is still established, so warm() is True
    async with aai.AsyncDictationTranscriber() as transcriber:
        assert await transcriber.warm() is True


async def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    # Given the dictation host is unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    # When warming, Then the failure is swallowed and reported as False
    async with aai.AsyncDictationTranscriber() as transcriber:
        assert await transcriber.warm() is False


async def test_api_key_constructor_builds_own_client(httpx_mock: HTTPXMock):
    # Given a transcriber constructed with an explicit key
    _mock_ok(httpx_mock)

    # When transcribing
    async with aai.AsyncDictationTranscriber(api_key="per-call-key") as transcriber:
        await transcriber.transcribe(b"RIFFfake")

    # Then that key authenticates the request
    assert httpx_mock.get_requests()[0].headers["authorization"] == "per-call-key"


async def test_rejects_sync_transcription_config():
    # Given the sync API's config, a different non-interchangeable type
    # When constructing, Then the mistake is named
    with pytest.raises(TypeError, match="expects DictationConfig"):
        aai.AsyncDictationTranscriber(config=aai.SyncTranscriptionConfig())


async def test_context_manager_closes_owned_client():
    # Given a transcriber that created its own client
    async with aai.AsyncDictationTranscriber() as transcriber:
        assert isinstance(transcriber, aai.AsyncDictationTranscriber)
        assert not transcriber.client.http_client.is_closed

    # Then leaving the block closes the owned connection pool
    assert transcriber.client.http_client.is_closed


async def test_aclose_leaves_shared_client_open():
    # Given a transcriber built on a caller-owned client
    async with aai.AsyncClient(settings=aai.settings) as client:
        transcriber = aai.AsyncDictationTranscriber(client=client)

        # When closing the transcriber
        await transcriber.aclose()

        # Then the shared pool stays open — its creator closes it
        assert not client.http_client.is_closed

    assert client.http_client.is_closed
