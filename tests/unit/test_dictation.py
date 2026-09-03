import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

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


def test_public_names_are_the_dictation_models_classes():
    # Given the module the dictation models live in
    from assemblyai.dictation.v1 import models

    # Then the top-level and versioned names are that module's classes
    assert aai.DictationConfig is models.DictationConfig
    assert aai.DictationError is models.DictationError
    assert aai.DictationResponse is models.DictationResponse
    assert aai.DictationWord is models.DictationWord


def test_transcribe_bytes_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing raw audio bytes
    result = aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the response is parsed into a DictationResponse
    assert isinstance(result, aai.DictationResponse)
    assert result.text == "patient reports mild headache"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[1].text == "reports"
    assert result.words[1].confidence == 0.95
    assert result.confidence == 0.92
    assert result.audio_duration_ms == 400
    assert result.request_time_ms == 243.7
    assert result.sync_time_ms == 180.2


def test_transcribe_posts_to_dictation_base_url(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the request goes to the dictation host, not the sync host
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == f"{aai.settings.dictation_base_url}/v1/transcribe"


def test_transcribe_sends_raw_api_key_and_no_model_header(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes")

    # Then the API key is sent raw, with no Bearer prefix
    request = httpx_mock.get_requests()[0]
    assert request.headers["authorization"] == "test"
    # And the Dictation API picks its own model, so no routing header is sent
    assert "X-AAI-Model" not in request.headers


def test_transcribe_sends_config_part_before_audio_part(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing with a config
    config = aai.DictationConfig(language_codes=["en"])
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config part precedes the audio part in the multipart body —
    # the server reads the config before it starts consuming the audio stream
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' in body
    assert b'name="audio"' in body
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"Content-Type: application/json" in body


def test_transcribe_omits_config_part_when_empty(httpx_mock: HTTPXMock):
    # Given a default config with nothing set
    _mock_ok(httpx_mock)

    # When transcribing, Then no config part is sent
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes")
    body = httpx_mock.get_requests()[0].read()
    assert b'name="config"' not in body


def test_config_json_excludes_unset_fields(httpx_mock: HTTPXMock):
    # Given a config that sets only llm_instruction
    _mock_ok(httpx_mock)
    config = aai.DictationConfig(llm_instruction="Format this as a SOAP note.")

    # When transcribing
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config JSON carries that field and omits the unset ones
    body = httpx_mock.get_requests()[0].read()
    assert b"Format this as a SOAP note." in body
    assert b'"language_codes"' not in body
    assert b'"sample_rate"' not in body
    assert b'"keyterms_prompt"' not in body


def test_transcribe_sends_keyterms_prompt_and_language_codes(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing with keyterms and languages
    config = aai.DictationConfig(
        keyterms_prompt=["AssemblyAI", "  Universal  ", ""],
        language_codes=["en", "es"],
    )
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes", config=config)

    # Then the config part carries both, with keyterms normalized
    body = httpx_mock.get_requests()[0].read()
    assert b'"AssemblyAI"' in body
    assert b'"Universal"' in body  # whitespace stripped, empty term dropped
    assert b'"language_codes"' in body
    assert b'"es"' in body


def test_transcribe_pcm_sends_pcm_part_and_rate(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing bytes with sample_rate + channels (raw PCM)
    config = aai.DictationConfig(sample_rate=16000, channels=1)
    aai.DictationTranscriber().transcribe(b"\x00\x01" * 100, config=config)

    # Then the audio part is PCM and the config carries rate + channels
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


def test_transcribe_pcm_without_channels_raises():
    # Given a config with sample_rate but no channels (partial PCM intent)
    config = aai.DictationConfig(sample_rate=16000)

    # When transcribing, Then it fails locally before any request
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.DictationTranscriber().transcribe(b"\x00\x01" * 100, config=config)


def test_transcribe_pcm_path_requires_rate_and_channels(tmp_path):
    # Given a .pcm file and a config with neither rate nor channels
    audio_file = tmp_path / "note.pcm"
    audio_file.write_bytes(b"\x00\x01" * 100)

    # When transcribing, Then the extension alone routes it as PCM and the
    # missing fields are caught locally
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.DictationTranscriber().transcribe(str(audio_file))


def test_transcribe_maps_mp3_extension_to_audio_mpeg(httpx_mock: HTTPXMock, tmp_path):
    # Given a local MP3 file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When transcribing the path
    aai.DictationTranscriber().transcribe(str(audio_file))

    # Then the audio part carries the MP3 Content-Type and the file's name
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/mpeg" in body
    assert b'filename="note.mp3"' in body


def test_transcribe_maps_wav_extension_to_audio_wav(httpx_mock: HTTPXMock, tmp_path):
    # Given a local WAV file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.wav"
    audio_file.write_bytes(b"RIFFfake-wav-bytes")

    # When transcribing the path
    result = aai.DictationTranscriber().transcribe(str(audio_file))

    # Then it succeeds and ships a WAV audio part
    assert result.text == "patient reports mild headache"
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/wav" in body


def test_transcribe_defaults_unknown_extension_to_wav(httpx_mock: HTTPXMock):
    # Given raw bytes with no filename to read an extension from
    _mock_ok(httpx_mock)

    # When transcribing, Then the audio part falls back to WAV
    aai.DictationTranscriber().transcribe(b"RIFFfake-wav-bytes")
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/wav" in body
    assert b'filename="audio.wav"' in body


def test_transcribe_rejects_url():
    # Given an http URL as input
    transcriber = aai.DictationTranscriber()

    # When transcribing, Then it is rejected with a pointer to Transcriber,
    # and the message names this client
    with pytest.raises(ValueError, match="DictationTranscriber does not accept URLs"):
        transcriber.transcribe("https://example.com/audio.wav")


def test_final_text_prefers_llm_response(httpx_mock: HTTPXMock):
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
    result = aai.DictationTranscriber().transcribe(b"RIFFfake", config=config)

    # Then final_text is the rewrite and the raw transcript is still there
    assert result.final_text == "S: Mild headache."
    assert result.text == "patient reports mild headache"


def test_final_text_falls_back_to_text(httpx_mock: HTTPXMock):
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
    result = aai.DictationTranscriber().transcribe(b"RIFFfake")

    # Then final_text falls back to the raw transcript
    assert result.final_text == "patient reports mild headache"
    assert result.llm_error == "llm timed out"


def test_response_ignores_unknown_fields(httpx_mock: HTTPXMock):
    # Given a server that added a field this SDK version predates
    response = dict(_OK_RESPONSE)
    response["auth_time_ms"] = 1.4
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When transcribing, Then the extra key is ignored instead of failing
    result = aai.DictationTranscriber().transcribe(b"RIFFfake")
    assert result.text == "patient reports mild headache"


def test_config_rejects_unknown_fields():
    # Given an option the Dictation API does not accept
    # When building the config, Then it is rejected rather than dropped
    with pytest.raises(ValueError):
        aai.DictationConfig(prompt="Transcribe verbatim.")


def test_keyterms_prompt_too_long_raises():
    # Given a keyterms_prompt exceeding the 2048-char cap
    # When building the config, Then validation fails immediately
    with pytest.raises(ValueError, match="keyterms_prompt exceeds"):
        aai.DictationConfig(keyterms_prompt=["x" * 3000])


def test_llm_instruction_too_long_raises():
    # Given an llm_instruction exceeding the 2048-char cap
    # When building the config, Then validation fails immediately
    with pytest.raises(ValueError):
        aai.DictationConfig(llm_instruction="x" * 3000)


def test_error_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given a server rejecting the audio with an {error, error_code} body
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=415,
        json={"error": "unsupported media type", "error_code": "bad_audio"},
    )

    # When transcribing, Then a DictationError carries code + status
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 415
    assert error.error_code == "bad_audio"
    assert "unsupported media type" in str(error)
    # And it is not the sync API's error class
    assert not isinstance(error, aai.SyncTranscriptError)


def test_problem_details_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given the server rejects oversized audio with a problem-details body
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=413,
        json={"status": 413, "title": "Audio Too Large", "detail": "too long"},
    )

    # When transcribing, Then the snake_cased title becomes the error_code
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


def test_detail_only_envelope_on_invalid_key(httpx_mock: HTTPXMock):
    # Given an invalid key, which the Dictation API answers 404
    httpx_mock.add_response(
        url=TRANSCRIBE_URL,
        method="POST",
        status_code=404,
        json={"detail": "Invalid API key"},
    )

    # When transcribing, Then the detail becomes the message and error_code
    # stays absent
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 404
    assert error.error_code is None
    assert "Invalid API key" in str(error)


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
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe(b"RIFFfake")

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


def test_transcribe_async_returns_future(httpx_mock: HTTPXMock):
    # Given a mocked dictation endpoint
    _mock_ok(httpx_mock)

    # When transcribing on a worker thread
    with aai.DictationTranscriber() as transcriber:
        future = transcriber.transcribe_async(b"RIFFfake")
        result = future.result()

    # Then the future yields the parsed transcript
    assert result.text == "patient reports mild headache"


def test_warm_opens_connection(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    warmed = aai.DictationTranscriber().warm()

    # Then it returns True and probes the dictation warm route
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == WARM_URL
    assert request.method == "GET"
    assert "X-AAI-Model" not in request.headers


def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    # Given a warm route that the load balancer answers with a 404
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    # When warming, Then the socket is still established, so warm() is True
    assert aai.DictationTranscriber().warm() is True


def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    # Given the dictation host is unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    # When warming, Then the failure is swallowed and reported as False
    assert aai.DictationTranscriber().warm() is False


def test_api_key_constructor_builds_own_client(httpx_mock: HTTPXMock):
    # Given a transcriber constructed with an explicit key
    _mock_ok(httpx_mock)

    # When transcribing
    aai.DictationTranscriber(api_key="per-call-key").transcribe(b"RIFFfake")

    # Then that key authenticates the request
    assert httpx_mock.get_requests()[0].headers["authorization"] == "per-call-key"


def test_rejects_sync_transcription_config():
    # Given the sync API's config, a different non-interchangeable type
    # When constructing, Then the mistake is named
    with pytest.raises(TypeError, match="expects DictationConfig"):
        aai.DictationTranscriber(config=aai.SyncTranscriptionConfig())


def test_config_setter_validates_and_round_trips():
    # Given a transcriber and a new config
    transcriber = aai.DictationTranscriber()
    config = aai.DictationConfig(language_codes=["de"])

    # When assigning it
    transcriber.config = config

    # Then it is the transcriber's default
    assert transcriber.config is config

    # And a wrong type is rejected
    with pytest.raises(TypeError, match="expects DictationConfig"):
        transcriber.config = aai.SyncTranscriptionConfig()


def test_context_manager_returns_self_and_closes():
    # Given a transcriber used as a context manager
    with aai.DictationTranscriber() as transcriber:
        # Then the bound value is the transcriber itself
        assert isinstance(transcriber, aai.DictationTranscriber)

    # And leaving the block shuts the worker pool down
    assert transcriber._executor._shutdown is True


def test_dictation_settings_defaults():
    # Given default settings
    settings = aai.Settings()

    # Then the dictation host and the documented 300s timeout are in place
    assert settings.dictation_base_url == "https://dictation.assemblyai.com"
    assert settings.dictation_http_timeout == 300.0
