import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

aai.settings.api_key = "test"

TRANSCRIBE_URL = f"{aai.settings.sync_base_url}/transcribe"

_OK_RESPONSE = {
    "text": "hello world",
    "words": [
        {"text": "hello", "start_ms": 0, "end_ms": 200, "confidence": 0.9},
        {"text": "world", "start_ms": 220, "end_ms": 400, "confidence": 0.95},
    ],
    "confidence": 0.92,
    "audio_duration_ms": 400,
    "inference_time_ms": 12.5,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
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
    assert result.words[0].start_ms == 0
    assert result.words[1].text == "world"


def test_transcribe_sends_model_header_and_wav_part(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with the default config
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the request routes via X-AAI-Model and ships a WAV audio part
    request = httpx_mock.get_requests()[0]
    assert request.headers["X-AAI-Model"] == "u3-sync-pro"
    body = request.read()
    assert b'name="audio"' in body
    assert b"Content-Type: audio/wav" in body
    # And no config part is sent when the config is empty
    assert b'name="config"' not in body


def test_transcribe_sends_prompt_and_word_boost(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a prompt and word_boost
    config = aai.SyncTranscriptionConfig(
        prompt="Transcribe verbatim.",
        word_boost=["AssemblyAI", "  Lemur  ", ""],
    )
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then a config JSON part carries the prompt and normalized word_boost
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


def test_conversation_context_rejects_too_many_chars():
    # Given conversation_context whose total length exceeds the cap,
    # When/Then constructing the config raises a validation error
    with pytest.raises(Exception):
        aai.SyncTranscriptionConfig(conversation_context=["a" * 5000])


def test_transcribe_sends_single_language_code(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a single language code
    config = aai.SyncTranscriptionConfig(language_code="es")
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries language_code as a bare string
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b'"language_code": "es"' in body


def test_transcribe_sends_language_code_list(httpx_mock: HTTPXMock):
    # Given a mocked sync endpoint
    _mock_ok(httpx_mock)

    # When transcribing with multiple language codes
    config = aai.SyncTranscriptionConfig(language_code=["en", "es"])
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON part carries language_code as a list
    body = httpx_mock.get_requests()[0].read()
    assert b'"language_code"' in body
    assert b'"en"' in body
    assert b'"es"' in body


def test_default_config_omits_language_code(httpx_mock: HTTPXMock):
    # Given a default config (no language specified)
    _mock_ok(httpx_mock)

    # When transcribing, Then no config part is sent and the server defaults
    # the language to English
    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' not in body


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


def test_word_boost_too_long_raises():
    # Given a word_boost exceeding the 2048-char cap
    # When building the config, Then validation fails immediately
    with pytest.raises(ValueError, match="word_boost exceeds"):
        aai.SyncTranscriptionConfig(word_boost=["x" * 3000])


def test_error_envelope_maps_to_sync_transcript_error(httpx_mock: HTTPXMock):
    # Given the server rejects oversized audio
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
        json={"detail": "Too many requests"},
        headers={"Retry-After": "5"},
    )

    # When transcribing, Then retry_after is parsed and error_code is absent
    with pytest.raises(aai.SyncTranscriptError) as exc_info:
        aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code is None
    assert error.retry_after == 5


def test_default_model_is_u3_sync_pro():
    # Given a default config
    # When inspecting the model
    # Then it is the sync U3-Pro identifier
    assert aai.SyncTranscriptionConfig().model == "u3-sync-pro"
    assert aai.SyncSpeechModel.u3_sync_pro.value == "u3-sync-pro"
