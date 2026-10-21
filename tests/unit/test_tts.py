"""Tests for the Text-to-Speech HTTP client.

Every entry point posts one JSON body to `/v1/speech` on the TTS host and
reads audio back; the response headers carry the metadata.
"""

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

aai.settings.api_key = "test"

SPEECH_URL = f"{aai.settings.tts_base_url}/v1/speech"

_AUDIO = b"\x00\x01" * 480  # 20 ms of 24 kHz s16le

_OK_HEADERS = {
    "content-type": "audio/pcm;rate=24000;channels=1",
    "x-request-id": "0199c6f1-2d0b-7c3e-9a41-5b6f0e1d2c3a",
    "x-tts-region": "us-east-1",
    "x-tts-model": "assemblyai-tts-1",
    "x-tts-voice": "jane",
    "x-tts-language": "english",
    "x-tts-encoding": "pcm_s16le",
    "x-tts-sample-rate": "24000",
    "x-tts-container": "raw",
    "x-tts-billed-characters": "12",
}

_ERROR_BODY = {
    "code": "invalid_parameter",
    "message": "sample_rate: sample_rate 12345 not in supported set",
    "param": "sample_rate",
    "request_id": "req-1",
    "errors": [{"message": "not in supported set", "param": "sample_rate"}],
    "retryable": False,
    "retry_after_seconds": None,
    "error": "sample_rate: sample_rate 12345 not in supported set",
}


def _mock_ok(httpx_mock: HTTPXMock, **headers: str) -> None:
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.OK,
        content=_AUDIO,
        headers={**_OK_HEADERS, **headers},
    )


def _sent_body(httpx_mock: HTTPXMock) -> dict:
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    return json.loads(requests[0].content)


def test_synthesize_posts_a_buffered_request_to_the_tts_host(httpx_mock: HTTPXMock):
    # GIVEN a TTS host that answers audio
    _mock_ok(httpx_mock, **{"x-tts-audio-duration-ms": "20"})

    # WHEN a text is synthesized with no options
    response = aai.TextToSpeech().synthesize("Hello there.")

    # THEN one buffered request went to the TTS host with the raw key
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == SPEECH_URL
    assert request.headers["authorization"] == "test"
    assert _sent_body(httpx_mock) == {"text": "Hello there.", "stream": False}

    # AND the audio and the headers are on the response
    assert response.audio == _AUDIO
    assert response.request_id == "0199c6f1-2d0b-7c3e-9a41-5b6f0e1d2c3a"
    assert response.model == "assemblyai-tts-1"
    assert response.voice == "jane"
    assert response.language == "english"
    assert response.encoding == "pcm_s16le"
    assert response.sample_rate == 24000
    assert response.container == "raw"
    assert response.bit_rate is None
    assert response.content_type == "audio/pcm;rate=24000;channels=1"
    assert response.billed_characters == 12
    assert response.audio_duration_ms == 20


def test_config_fields_go_in_the_body_and_unset_ones_stay_out(httpx_mock: HTTPXMock):
    # GIVEN a default config on the client and a per-call config
    _mock_ok(httpx_mock)
    tts = aai.TextToSpeech(config=aai.TTSConfig(voice="alba", language="english"))

    # WHEN a per-call config is given
    tts.synthesize(
        "Hola.",
        aai.TTSConfig(
            voice_id="aai-juan",
            model_id="assemblyai-tts-1",
            sample_rate=8000,
            encoding=aai.TTSEncoding.pcm_mulaw,
            container=aai.TTSContainer.wav,
        ),
    )

    # THEN it replaces the default, enums serialize to their values
    assert _sent_body(httpx_mock) == {
        "text": "Hola.",
        "voice_id": "aai-juan",
        "model_id": "assemblyai-tts-1",
        "sample_rate": 8000,
        "encoding": "pcm_mulaw",
        "container": "wav",
        "stream": False,
    }


def test_default_config_is_used_when_no_per_call_config(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    aai.TextToSpeech(config=aai.TTSConfig(voice="alba")).synthesize("Hi.")

    assert _sent_body(httpx_mock) == {"text": "Hi.", "voice": "alba", "stream": False}


def test_stream_yields_audio_as_it_arrives_and_reports_the_headers(
    httpx_mock: HTTPXMock,
):
    # GIVEN a streamed answer
    _mock_ok(httpx_mock)

    # WHEN a text is streamed
    with aai.TextToSpeech().stream("Hello there.") as speech:
        chunks = list(speech)

    # THEN the request asked for a streamed response and the bytes came back
    assert _sent_body(httpx_mock) == {"text": "Hello there.", "stream": True}
    assert b"".join(chunks) == _AUDIO
    assert speech.info.model == "assemblyai-tts-1"
    assert speech.info.sample_rate == 24000
    assert speech.info.audio_duration_ms is None


def test_stream_read_collects_the_whole_audio(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    speech = aai.TextToSpeech().stream("Hello there.")
    try:
        assert speech.read() == _AUDIO
    finally:
        speech.close()


def test_error_envelope_becomes_a_tts_error(httpx_mock: HTTPXMock):
    # GIVEN a refusal with the API's JSON envelope
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.UNPROCESSABLE_ENTITY,
        json=_ERROR_BODY,
        headers={"x-request-id": "req-1"},
    )

    # WHEN the request is made
    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.", aai.TTSConfig(sample_rate=12345))

    # THEN every field of the envelope is on the error
    err = exc_info.value
    assert isinstance(err, aai.AssemblyAIError)
    assert str(err) == _ERROR_BODY["message"]
    assert err.status_code == 422
    assert err.error_code == "invalid_parameter"
    assert err.param == "sample_rate"
    assert err.request_id == "req-1"
    assert err.retryable is False
    assert err.retry_after is None
    assert err.errors == _ERROR_BODY["errors"]


def test_stream_raises_the_same_error_before_any_audio(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.TOO_MANY_REQUESTS,
        json={
            **_ERROR_BODY,
            "code": "rate_limited",
            "message": "Too many requests",
            "param": None,
            "errors": None,
            "retryable": True,
            "retry_after_seconds": 17,
        },
        headers={"retry-after": "17"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().stream("Hi.")

    assert exc_info.value.error_code == "rate_limited"
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after == 17


def test_retry_after_header_is_read_when_the_body_has_no_hint(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"code": "at_capacity", "message": "No room", "retryable": True},
        headers={"retry-after": "2"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "at_capacity"
    assert exc_info.value.retry_after == 2


def test_non_json_error_reports_the_status_and_text(httpx_mock: HTTPXMock):
    # GIVEN an answer made before the request reached the API
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.BAD_GATEWAY,
        content=b"<html>502 Bad Gateway</html>",
        headers={"content-type": "text/html"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    err = exc_info.value
    assert err.status_code == 502
    assert err.error_code is None
    assert err.retryable is True
    assert "502" in str(err)
    assert "Bad Gateway" in str(err)


def test_tts_base_url_setting_picks_the_host(httpx_mock: HTTPXMock):
    # GIVEN the EU host in the settings
    eu_url = "https://tts.eu.assemblyai.com/v1/speech"
    httpx_mock.add_response(
        url=eu_url, method="POST", status_code=200, content=_AUDIO, headers=_OK_HEADERS
    )
    client = aai.Client(api_key="test")
    client.settings.tts_base_url = "https://tts.eu.assemblyai.com"

    # WHEN a request is made with that client
    aai.TextToSpeech(client=client).synthesize("Hallo.")

    # THEN it went to the EU host
    assert str(httpx_mock.get_requests()[0].url) == eu_url


def test_setting_config_to_none_restores_the_defaults(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    tts = aai.TextToSpeech(config=aai.TTSConfig(voice="alba"))
    tts.config = None  # type: ignore[assignment]

    tts.synthesize("Hi.")

    assert _sent_body(httpx_mock) == {"text": "Hi.", "stream": False}


def test_json_error_without_retryable_falls_back_to_the_status(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"message": "upstream unavailable"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.retryable is True
    assert exc_info.value.error_code is None


def test_voice_expiry_header_is_reported(httpx_mock: HTTPXMock):
    _mock_ok(
        httpx_mock, **{"x-tts-voice": "custom", "x-tts-voice-expires-at": "1791480600"}
    )

    response = aai.TextToSpeech().synthesize("Hi.", aai.TTSConfig(voice_id="aai-x"))

    assert response.voice == "custom"
    assert response.voice_expires_at == 1791480600


def test_encoding_takes_the_enum_or_a_string_alias(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    tts = aai.TextToSpeech()

    tts.synthesize("Hi.", aai.TTSConfig(encoding="mulaw"))
    tts.synthesize("Hi.", aai.TTSConfig(encoding=aai.TTSEncoding.pcm_alaw))

    bodies = [json.loads(r.content) for r in httpx_mock.get_requests()]
    assert bodies[0]["encoding"] == "mulaw"
    assert bodies[1]["encoding"] == "pcm_alaw"


def test_synthesize_and_stream_use_their_own_timeouts(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    client = aai.Client(api_key="test")
    client.settings.tts_http_timeout = 7.0
    client.settings.tts_synthesize_http_timeout = 99.0
    tts = aai.TextToSpeech(client=client)

    tts.synthesize("Hi.")
    with tts.stream("Hi.") as speech:
        speech.read()

    timeouts = [r.extensions["timeout"]["read"] for r in httpx_mock.get_requests()]
    assert timeouts == [99.0, 7.0]


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        aai.TTSConfig(voice="jane", speed=2)  # type: ignore[call-arg]


def test_wrong_config_type_is_a_type_error():
    with pytest.raises(TypeError):
        aai.TextToSpeech(config=aai.DictationConfig())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        aai.TextToSpeech().synthesize("Hi.", aai.SyncTranscriptionConfig())  # type: ignore[arg-type]


def test_text_must_be_a_str():
    with pytest.raises(TypeError):
        aai.TextToSpeech().synthesize(b"Hi.")  # type: ignore[arg-type]


def test_swapped_text_and_config_is_named():
    with pytest.raises(TypeError, match="text comes first"):
        aai.TextToSpeech().stream(aai.TTSConfig(voice="jane"), "Hi.")  # type: ignore[arg-type]


def test_a_dict_config_is_a_type_error_not_an_attribute_error():
    with pytest.raises(TypeError, match="TTSConfig"):
        aai.TextToSpeech().synthesize("Hi.", {"voice": "jane"})  # type: ignore[arg-type]
