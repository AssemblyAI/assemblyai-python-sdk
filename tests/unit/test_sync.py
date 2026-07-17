import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

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


def test_transcribe_bytes_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing raw audio bytes
    result = aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the response is parsed into a SyncTranscriptResponse
    assert isinstance(result, aai.SyncTranscriptResponse)
    assert result.text == "hello world"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[0].start == 0
    assert result.words[0].end == 200
    assert result.words[1].text == "world"
    assert result.request_time_ms == 243.7


def test_transcribe_parses_response_without_request_time(httpx_mock: HTTPXMock):
    # Given a server response that predates the request_time_ms field
    response = {k: v for k, v in _OK_RESPONSE.items() if k != "request_time_ms"}
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When transcribing
    result = aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then request_time_ms is None instead of a parse failure
    assert result.request_time_ms is None


def test_transcribe_sends_model_header_and_wav_part(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with the default config
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the request routes via X-AAI-Model and ships a WAV audio part
    request = httpx_mock.get_requests()[0]
    assert request.headers["X-AAI-Model"] == "universal-3-5-pro"
    body = request.read()
    assert b'name="audio"' in body
    assert b"Content-Type: audio/wav" in body
    # And no config part is sent when the config is empty
    assert b'name="config"' not in body


def test_transcribe_sends_prompt_and_keyterms_prompt(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a prompt and keyterms_prompt
    config = aai.SyncTranscriptionConfig(
        prompt="Transcribe verbatim.",
        keyterms_prompt=["AssemblyAI", "  Lemur  ", ""],
    )
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then a config JSON part carries the prompt and normalized keyterms_prompt
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b"Transcribe verbatim." in body
    assert b'"AssemblyAI"' in body
    assert b'"Lemur"' in body  # whitespace stripped, empty term dropped
    # And the routing model is never placed in the body
    assert b'"model"' not in body


def test_transcribe_sends_conversation_context_list(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with prior conversation turns (oldest first)
    config = aai.SyncTranscriptionConfig(
        conversation_context=[
            "I'd like to book a flight to Denver.",
            "  Sure, what date were you thinking?  ",
            "",
        ],
    )
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries the turns, stripped with empties dropped
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b'"conversation_context"' in body
    assert b"I'd like to book a flight to Denver." in body
    assert b"Sure, what date were you thinking?" in body


def test_transcribe_coerces_conversation_context_string(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When conversation_context is a bare string (single prior turn)
    config = aai.SyncTranscriptionConfig(
        conversation_context="Sure, what date were you thinking?"
    )

    # Then it is normalized to a one-turn list
    assert config.conversation_context == ["Sure, what date were you thinking?"]

    # And it ships as a JSON array in the config part
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)
    body = httpx_mock.get_requests()[0].read()
    assert b'"conversation_context"' in body
    assert b'"Sure, what date were you thinking?"' in body


def test_conversation_context_trims_oldest_turns_over_char_cap():
    # Given conversation_context whose total length exceeds the 4096-char cap
    config = aai.SyncTranscriptionConfig(conversation_context=["a" * 3000, "b" * 3000])

    # Then the oldest turn is dropped and the most recent turn is kept
    assert config.conversation_context == ["b" * 3000]


def test_conversation_context_trims_oldest_turns_over_turn_cap():
    # Given conversation_context with more than 100 turns
    turns = [f"turn {i}" for i in range(120)]
    config = aai.SyncTranscriptionConfig(conversation_context=turns)

    # Then it is trimmed to the 100 most recent turns, oldest dropped first
    assert config.conversation_context == turns[20:]


def test_conversation_context_empties_when_single_turn_over_char_cap():
    # Given a single turn that alone exceeds the character cap
    config = aai.SyncTranscriptionConfig(conversation_context=["a" * 5000])

    # Then the context trims to nothing rather than raising
    assert config.conversation_context is None


def test_transcribe_sends_single_language_codes(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a single language code
    config = aai.SyncTranscriptionConfig(language_codes=["es"])
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries language_codes as a one-element list
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b'"language_codes": ["es"]' in body


def test_transcribe_sends_language_codes_list(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with multiple language codes
    config = aai.SyncTranscriptionConfig(language_codes=["en", "es"])
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries language_codes as a list
    body = httpx_mock.get_requests()[0].read()
    assert b'"language_codes"' in body
    assert b'"en"' in body
    assert b'"es"' in body


def test_language_codes_rejects_bare_string():
    # Given a bare string instead of a list
    # When building the config, Then validation fails — language_codes only
    # accepts a list of codes
    with pytest.raises(Exception):
        aai.SyncTranscriptionConfig(language_codes="es")


def test_default_config_omits_language_code(httpx_mock: HTTPXMock):
    # Given a default config (no language specified)
    _mock_ok(httpx_mock)

    # When transcribing, Then no config part is sent and the server defaults
    # the language to English
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' not in body


def test_transcribe_sends_timestamps_flag(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with timestamps opted in
    config = aai.SyncTranscriptionConfig(timestamps=True)
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries the flag
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b'"timestamps": true' in body


def test_transcribe_parses_words_without_timestamps(httpx_mock: HTTPXMock):
    # Given a response whose words carry no start/end (timestamps not
    # requested — the server omits the fields rather than sending null)
    response = dict(_OK_RESPONSE)
    response["words"] = [
        {"text": "hello", "confidence": 0.9},
        {"text": "world", "confidence": 0.95},
    ]
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When transcribing
    result = aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the words parse with None timings instead of failing
    assert result.words[0].text == "hello"
    assert result.words[0].start is None
    assert result.words[0].end is None
    assert result.words[1].confidence == 0.95


def test_transcribe_pcm_sends_pcm_part_and_rate(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with sample_rate + channels (raw PCM)
    config = aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
    aai.SyncTranscriber().transcribe(b"\x00\x01" * 100, config=config)

    # Then the audio part is PCM and the config carries rate + channels
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


def test_transcribe_pcm_without_rate_raises():
    # Given a config with sample_rate but no channels (partial PCM intent)
    config = aai.SyncTranscriptionConfig(sample_rate=16000)

    # When transcribing, Then it fails locally before any request
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.SyncTranscriber().transcribe(b"\x00\x01" * 100, config=config)


def test_transcribe_rejects_url():
    # Given an http URL as input
    transcriber = aai.SyncTranscriber()

    # When transcribing, Then it is rejected with a pointer to Transcriber
    with pytest.raises(ValueError, match="does not accept URLs"):
        transcriber.transcribe("https://example.com/audio.wav")


def test_transcribe_path_input(httpx_mock: HTTPXMock, tmp_path):
    # Given a local WAV file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"RIFFfake-wav-bytes")

    # When transcribing the path
    result = aai.SyncTranscriber().transcribe(str(audio_file))

    # Then it succeeds and ships the file under its own name
    assert result.text == "hello world"
    body = httpx_mock.get_requests()[0].read()
    assert b'filename="call.wav"' in body


def test_keyterms_prompt_too_long_raises():
    # Given a keyterms_prompt exceeding the 2048-char cap
    # When building the config, Then validation fails immediately
    with pytest.raises(ValueError, match="keyterms_prompt exceeds"):
        aai.SyncTranscriptionConfig(keyterms_prompt=["x" * 3000])


def test_problem_details_envelope_maps_to_sync_transcript_error(
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
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


def test_legacy_error_envelope_maps_to_sync_transcript_error(
    httpx_mock: HTTPXMock,
):
    # Given a server still on the pre-problem-details envelope
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=413,
        json={"error_code": "audio_too_large", "message": "too long"},
    )

    # When transcribing, Then a SyncTranscriptError carries code + status
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


def test_rate_limit_surfaces_retry_after(httpx_mock: HTTPXMock):
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
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


def test_legacy_detail_only_envelope(httpx_mock: HTTPXMock):
    # Given an auth-style body with only a detail field
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=401,
        json={"detail": "Invalid API key"},
    )

    # When transcribing, Then the detail becomes the message and error_code
    # stays absent
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 401
    assert error.error_code is None
    assert "Invalid API key" in str(error)


def test_default_model_is_universal_3_5_pro():
    # Given a default config
    # When inspecting the model
    # Then it is universal-3-5-pro
    assert aai.SyncTranscriptionConfig().model == "universal-3-5-pro"
    assert aai.SyncSpeechModel.universal_3_5_pro.value == "universal-3-5-pro"


def test_warm_opens_connection_with_model_header(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    warmed = aai.SyncTranscriber().warm()

    # Then it returns True and routes the probe via X-AAI-Model
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert request.url == WARM_URL
    assert request.method == "GET"
    assert request.headers["X-AAI-Model"] == "universal-3-5-pro"


def test_warm_uses_configured_model(httpx_mock: HTTPXMock):
    # Given a transcriber pinned to a specific model
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)
    config = aai.SyncTranscriptionConfig(model="some-other-model")

    # When warming
    aai.SyncTranscriber(config=config).warm()

    # Then the warm probe carries that model so it lands on the right backend
    assert httpx_mock.get_requests()[0].headers["X-AAI-Model"] == "some-other-model"


def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    # Given a warm route that the load balancer answers with a 404
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    # When warming, Then the socket is still established, so warm() is True
    assert aai.SyncTranscriber().warm() is True


def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    # Given the sync host is unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    # When warming, Then the failure is swallowed and reported as False
    assert aai.SyncTranscriber().warm() is False


def test_context_manager_returns_self_and_closes():
    # Given a transcriber used as a context manager
    with aai.SyncTranscriber() as transcriber:
        # Then the bound value is the transcriber itself
        assert isinstance(transcriber, aai.SyncTranscriber)

    # And leaving the block shuts the worker pool down
    assert transcriber._executor._shutdown is True


def test_keepalive_expiry_defaults_to_httpx_default():
    # Given a default config
    # When inspecting keepalive_expiry
    # Then it is None, leaving httpx's own default in place
    assert aai.Settings().keepalive_expiry is None


def test_client_accepts_custom_keepalive_expiry():
    # Given a client configured with a longer keepalive
    from assemblyai import client as client_mod

    # When constructed, Then it builds cleanly (the value reaches httpx.Limits)
    # and round-trips on settings
    client = client_mod.Client(
        settings=aai.Settings(api_key="k", keepalive_expiry=120.0)
    )
    assert client.settings.keepalive_expiry == 120.0
    assert client.http_client is not None
