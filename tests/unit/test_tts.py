"""Tests for the Text-to-Speech HTTP client.

Every entry point posts one JSON body to `/v1/speech` on the TTS host and
reads audio back; the response headers carry the metadata.
"""

import gc
import json
import os
import re
import ssl
import threading
import time
import warnings
import weakref
from pathlib import Path
from typing import Iterator, List

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai import client as client_module
from assemblyai.tts.v1 import api
from tests.unit.tts_http_server import serving

aai.settings.api_key = "test"

SPEECH_URL = f"{aai.settings.tts_base_url}/v1/speech"

_AUDIO = b"\x00\x01" * 480  # 20 ms of 24 kHz s16le

_OK_HEADERS = {
    "content-type": "audio/pcm;rate=24000;channels=1",
    "x-request-id": "0199c6f1-2d0b-7c3e-9a41-5b6f0e1d2c3a",
    "x-tts-region": "us-east-1",
    "x-tts-model": "assemblyai-tts-1",
    "x-tts-voice-id": "aai-jane",
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


def _mock_error(httpx_mock: HTTPXMock, status: int, body: dict, **headers: str) -> None:
    httpx_mock.add_response(
        url=SPEECH_URL, method="POST", status_code=status, json=body, headers=headers
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
    assert response.region == "us-east-1"
    assert response.model == "assemblyai-tts-1"
    assert response.voice_id == "aai-jane"
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
    tts = aai.TextToSpeech(
        config=aai.TTSConfig(voice_id="aai-alba", language="english")
    )

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
    aai.TextToSpeech(config=aai.TTSConfig(voice_id="aai-alba")).synthesize("Hi.")

    assert _sent_body(httpx_mock) == {
        "text": "Hi.",
        "voice_id": "aai-alba",
        "stream": False,
    }


def test_voice_id_falls_back_to_x_tts_voice_until_the_api_sends_the_id(
    httpx_mock: HTTPXMock,
):
    # GIVEN a server that still sends X-TTS-Voice only
    legacy = {k: v for k, v in _OK_HEADERS.items() if k != "x-tts-voice-id"}
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        content=_AUDIO,
        headers={**legacy, "x-tts-voice": "jane"},
    )
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        content=_AUDIO,
        headers={**legacy, "x-tts-voice": "custom"},
    )
    tts = aai.TextToSpeech()

    # THEN a built-in voice's name becomes its slug, and `custom` the request's voice_id
    assert tts.synthesize("Hi.").voice_id == "aai-jane"
    cloned = tts.synthesize("Hi.", aai.TTSConfig(voice_id="aai-isaiah-2"))
    assert cloned.voice_id == "aai-isaiah-2"


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
    assert speech.info.voice_id == "aai-jane"
    assert speech.info.sample_rate == 24000
    assert speech.info.audio_duration_ms is None


def test_stream_sends_nothing_until_entered_and_raises_there(httpx_mock: HTTPXMock):
    _mock_error(
        httpx_mock, 422, {**_ERROR_BODY, "code": "voice_not_found", "param": "voice_id"}
    )

    speech = aai.TextToSpeech().stream("Hi.", aai.TTSConfig(voice_id="aai-x"))
    assert httpx_mock.get_requests() == []
    with pytest.raises(RuntimeError, match="enter or read"):
        speech.info

    with pytest.raises(aai.TTSError) as exc_info:
        with speech:
            pass
    assert exc_info.value.error_code == "voice_not_found"
    speech.close()  # a never-opened stream closes quietly


def test_stream_can_be_iterated_without_a_context_manager(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    speech = aai.TextToSpeech().stream("Hi.")
    chunks = list(speech)
    speech.close()

    assert b"".join(chunks) == _AUDIO
    assert speech.info.voice_id == "aai-jane"


def test_stream_read_collects_the_whole_audio(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    speech = aai.TextToSpeech().stream("Hello there.")
    try:
        assert speech.read() == _AUDIO
    finally:
        speech.close()


class _CutStream(httpx.SyncByteStream):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield _AUDIO[:100]
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.ReadError("connection reset"),
        httpx.ReadTimeout("no audio for 30 s"),
    ],
)
def test_a_body_cut_after_the_first_audio_is_a_retryable_tts_error(error):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers=_OK_HEADERS, stream=_CutStream(error)
        )
    )
    with httpx.Client(transport=transport) as http:
        speech = api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        with speech, pytest.raises(aai.TTSError) as exc_info:
            list(speech)

    err = exc_info.value
    assert err.retryable is True
    assert err.status_code == 200
    assert err.error_code is None
    assert err.request_id == _OK_HEADERS["x-request-id"]
    assert err.__cause__ is error


class _RecordingStream(httpx.SyncByteStream):
    closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield _AUDIO[:100]
        yield _AUDIO[100:]

    def close(self) -> None:
        self.closed = True


def _transport(body: httpx.SyncByteStream, status: int = 200) -> httpx.MockTransport:
    headers = _OK_HEADERS if status == 200 else {"content-type": "application/json"}
    return httpx.MockTransport(
        lambda request: httpx.Response(status, headers=headers, stream=body)
    )


def test_leaving_the_with_block_early_releases_the_connection():
    body = _RecordingStream()
    with httpx.Client(transport=_transport(body)) as http:
        with api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        ) as speech:
            next(iter(speech))  # a barge-in after the first chunk
        assert body.closed


def test_close_releases_a_stream_read_outside_a_with_block():
    body = _RecordingStream()
    with httpx.Client(transport=_transport(body)) as http:
        speech = api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        speech.read()
        assert body.closed  # a fully read stream is released by httpx itself
        speech.close()  # and a second close is harmless


class _JsonBody(httpx.SyncByteStream):
    closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield json.dumps({"code": "at_capacity", "message": "No room"}).encode()

    def close(self) -> None:
        self.closed = True


def test_a_refused_stream_releases_its_connection():
    body = _JsonBody()
    with httpx.Client(transport=_transport(body, status=503)) as http:
        speech = api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        with pytest.raises(aai.TTSError) as exc_info:
            speech.read()
    assert exc_info.value.error_code == "at_capacity"
    assert body.closed


class _CutJsonBody(httpx.SyncByteStream):
    closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield b'{"code": '
        raise httpx.ReadError("connection reset")

    def close(self) -> None:
        self.closed = True


def test_a_refused_stream_releases_its_connection_if_its_body_fails():
    # httpx closes a fully read body itself; a cut body is released only by
    # the stream. Holding the error keeps `cm` alive, so GC cannot do it.
    body = _CutJsonBody()
    with httpx.Client(transport=_transport(body, status=503)) as http:
        speech = api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        with pytest.raises(httpx.ReadError) as exc_info:
            speech.read()
        assert body.closed
        assert "connection reset" in str(exc_info.value)


def test_a_client_built_from_api_key_is_closed_with_the_instance():
    with aai.TextToSpeech(api_key="k") as tts:
        http = tts.client.http_client
        assert not http.is_closed
    assert http.is_closed

    # the shared default client is never closed by an instance
    shared = aai.TextToSpeech()
    shared.close()
    assert not shared.client.http_client.is_closed


def test_error_body_is_kept_for_the_fields_some_codes_add(httpx_mock: HTTPXMock):
    _mock_error(
        httpx_mock,
        429,
        {
            "code": "rate_limited",
            "message": "slow down",
            "retryable": True,
            "limit": 100,
            "limit_window": "minute",
        },
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.body is not None
    assert exc_info.value.body["limit"] == 100
    assert exc_info.value.body["limit_window"] == "minute"


def test_config_is_a_native_model_of_the_installed_pydantic():
    import pydantic

    # On pydantic 2 without pydantic-settings the SDK's shared models run in
    # v1-compat mode; the TTS models must still be native v2 models.
    assert issubclass(aai.TTSConfig, pydantic.BaseModel)
    assert issubclass(aai.TTSResponse, pydantic.BaseModel)


def test_defining_the_models_raises_no_pydantic_warnings():
    import importlib

    from assemblyai.tts.v1 import models

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(models)


def test_response_repr_leaves_the_audio_out(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    response = aai.TextToSpeech().synthesize("Hi.")

    assert "audio=" not in repr(response)
    assert "voice_id='aai-jane'" in repr(response)
    assert response.audio == _AUDIO


def test_error_envelope_becomes_a_tts_error(httpx_mock: HTTPXMock):
    # GIVEN a refusal with the API's JSON envelope
    _mock_error(httpx_mock, 422, _ERROR_BODY, **{"x-request-id": "req-1"})

    # WHEN the request is made
    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.", aai.TTSConfig(sample_rate=12345))

    # THEN every field of the envelope is on the error
    err: aai.TTSError = exc_info.value
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
    _mock_error(
        httpx_mock,
        429,
        {
            **_ERROR_BODY,
            "code": "rate_limited",
            "message": "Too many requests",
            "param": None,
            "errors": None,
            "retryable": True,
            "retry_after_seconds": 17,
        },
        **{"retry-after": "17"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TextToSpeech().stream("Hi."):
            pass

    assert exc_info.value.error_code == "rate_limited"
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after == 17


def test_biometric_terms_refusal_and_auth_unavailable(httpx_mock: HTTPXMock):
    _mock_error(
        httpx_mock,
        403,
        {
            "code": "biometric_terms_not_accepted",
            "message": "Accept the biometric data terms in the dashboard.",
            "param": "voice_id",
            "retryable": False,
        },
    )
    _mock_error(
        httpx_mock,
        503,
        {"code": "auth_unavailable", "message": "Try again", "retryable": True},
        **{"retry-after": "1"},
    )
    tts = aai.TextToSpeech()

    with pytest.raises(aai.TTSError) as exc_info:
        tts.synthesize(
            "Hi.", aai.TTSConfig(voice_id="aai-x", extend_biometric_consent=True)
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_code == "biometric_terms_not_accepted"
    assert exc_info.value.retryable is False

    with pytest.raises(aai.TTSError) as exc_info:
        tts.synthesize("Hi.")
    assert exc_info.value.error_code == "auth_unavailable"
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after == 1


def test_retry_after_header_is_read_when_the_body_has_no_hint(httpx_mock: HTTPXMock):
    _mock_error(
        httpx_mock,
        503,
        {"code": "at_capacity", "message": "No room", "retryable": True},
        **{"retry-after": "2"},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "at_capacity"
    assert exc_info.value.retry_after == 2


def test_fractional_retry_after_is_rounded_up(httpx_mock: HTTPXMock):
    _mock_error(
        httpx_mock,
        429,
        {"code": "rate_limited", "message": "slow down", "retry_after_seconds": 1.2},
    )

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.retry_after == 2


def test_json_error_without_retryable_falls_back_to_the_status(httpx_mock: HTTPXMock):
    _mock_error(httpx_mock, 503, {"message": "upstream unavailable"})

    with pytest.raises(aai.TTSError) as exc_info:
        aai.TextToSpeech().synthesize("Hi.")

    assert exc_info.value.retryable is True
    assert exc_info.value.error_code is None


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

    timeouts = [r.extensions["timeout"] for r in httpx_mock.get_requests()]
    # synthesize: the read waits for the whole clip, connecting does not
    assert (timeouts[0]["read"], timeouts[0]["connect"]) == (99.0, 7.0)
    assert timeouts[1]["read"] == 7.0


def test_the_synthesize_read_timeout_outlasts_the_apis_request_limit():
    assert aai.Settings().tts_synthesize_http_timeout > 600


def test_setting_config_to_none_restores_the_defaults(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    tts = aai.TextToSpeech(config=aai.TTSConfig(voice_id="aai-alba"))
    tts.config = None  # type: ignore[assignment]

    tts.synthesize("Hi.")

    assert _sent_body(httpx_mock) == {"text": "Hi.", "stream": False}


def test_voice_expiry_header_is_reported(httpx_mock: HTTPXMock):
    _mock_ok(
        httpx_mock,
        **{"x-tts-voice-id": "aai-x", "x-tts-voice-expires-at": "1791480600"},
    )

    response = aai.TextToSpeech().synthesize("Hi.", aai.TTSConfig(voice_id="aai-x"))

    assert response.voice_id == "aai-x"
    assert response.voice_expires_at == 1791480600


def test_encoding_and_container_take_the_enum_or_a_string(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    tts = aai.TextToSpeech()

    tts.synthesize("Hi.", aai.TTSConfig(encoding="mulaw", container="wav"))
    tts.synthesize(
        "Hi.",
        aai.TTSConfig(
            encoding=aai.TTSEncoding.pcm_alaw, container=aai.TTSContainer.raw
        ),
    )

    bodies = [json.loads(r.content) for r in httpx_mock.get_requests()]
    assert (bodies[0]["encoding"], bodies[0]["container"]) == ("mulaw", "wav")
    assert (bodies[1]["encoding"], bodies[1]["container"]) == ("pcm_alaw", "raw")


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        aai.TTSConfig(voice_id="aai-jane", speed=2)  # type: ignore[call-arg]


def test_voice_is_not_a_config_field_and_the_error_names_voice_id():
    with pytest.raises(ValueError, match="voice_id"):
        aai.TTSConfig(voice="jane")  # type: ignore[call-arg]


def test_model_id_needs_no_protected_namespace_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aai.TTSConfig(model_id="assemblyai-tts-1")


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
        aai.TextToSpeech().stream(aai.TTSConfig(voice_id="aai-jane"), "Hi.")  # type: ignore[arg-type]


def test_a_dict_config_is_a_type_error_not_an_attribute_error():
    with pytest.raises(TypeError, match="TTSConfig"):
        aai.TextToSpeech().synthesize("Hi.", {"voice_id": "aai-jane"})  # type: ignore[arg-type]


def _keepalive_expiry(http: httpx.Client) -> float:
    """The idle time of `http`'s connection pool (httpcore's own field)."""

    return http._transport._pool._keepalive_expiry  # type: ignore[attr-defined]


def _ssl_context(http: httpx.Client) -> ssl.SSLContext:
    """The TLS context `http`'s connection pool connects with (httpcore's own field)."""

    return http._transport._pool._ssl_context  # type: ignore[attr-defined]


def test_the_tts_pool_keeps_idle_connections_120_s_by_default():
    assert aai.Settings().tts_keepalive_expiry == 120.0

    client = aai.Client(settings=aai.Settings(api_key="test"))
    pool = client.tts_http_client

    # A pool of its own, speaking HTTP/1.1, holding idle connections 120 s
    assert pool is not client.http_client
    assert pool is client.tts_http_client
    assert _keepalive_expiry(pool) == 120.0
    assert pool._transport._pool._http2 is False  # type: ignore[attr-defined]
    # while the pool of the other APIs keeps httpx's default
    assert _keepalive_expiry(client.http_client) == 5.0


def test_the_tts_keepalive_expiry_is_a_setting():
    settings = aai.Settings(api_key="test", tts_keepalive_expiry=30.0)
    assert _keepalive_expiry(aai.Client(settings=settings).tts_http_client) == 30.0

    with pytest.raises(ValueError):
        aai.Settings(tts_keepalive_expiry=-1.0)


def test_the_tts_pool_is_built_with_the_text_to_speech_not_with_the_client():
    client = aai.Client(api_key="test")
    assert client._tts_http_client is None  # a client that never speaks

    aai.TextToSpeech(client=client)

    # Built before the first request, which then does not wait for it
    assert client._tts_http_client is not None
    # and an instance on the shared default client builds its pool as well
    assert aai.TextToSpeech().client._tts_http_client is not None


def test_every_tts_pool_shares_one_verifying_tls_context():
    first = aai.TextToSpeech(api_key="k")
    second = aai.TextToSpeech(api_key="k")

    try:
        context = _ssl_context(first.client.tts_http_client)
        # One context for every TTS pool, so a new pool loads no certificates
        assert context is _ssl_context(second.client.tts_http_client)
        assert context is client_module._shared_tts_ssl_context()
        # httpx's own verifying context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        # The pool of the other APIs keeps a context of its own, as before
        assert _ssl_context(first.client.http_client) is not context
    finally:
        first.close()
        second.close()


def test_concurrent_first_calls_build_one_tls_context(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_module, "_tts_ssl_context", None)
    built: List[ssl.SSLContext] = []
    real_create = httpx.create_ssl_context

    def slow_create() -> ssl.SSLContext:
        time.sleep(0.05)  # widens the window two builders would race in
        built.append(real_create())
        return built[-1]

    monkeypatch.setattr(httpx, "create_ssl_context", slow_create)
    callers = 8
    start = threading.Barrier(callers)
    contexts: List[ssl.SSLContext] = []

    def take() -> None:
        start.wait()
        contexts.append(client_module._shared_tts_ssl_context())

    threads = [threading.Thread(target=take) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(built) == 1
    assert len(contexts) == callers
    assert all(context is built[0] for context in contexts)


def test_requests_go_through_the_tts_pool(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    client = aai.Client(api_key="test")
    tts = aai.TextToSpeech(client=client)
    # The pool of the other APIs is not needed
    client.http_client.close()

    tts.synthesize("Hi.")
    with tts.stream("Hi.") as speech:
        speech.read()

    assert len(httpx_mock.get_requests()) == 2
    # and responses still update last_response
    assert client.last_response is not None
    assert str(client.last_response.request.url) == SPEECH_URL


def test_concurrent_first_requests_build_one_tts_pool():
    client = aai.Client(api_key="test")
    callers = 8
    start = threading.Barrier(callers)
    pools: List[httpx.Client] = []

    def take() -> None:
        start.wait()
        pools.append(client.tts_http_client)

    threads = [threading.Thread(target=take) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(pools) == callers
    assert all(pool is pools[0] for pool in pools)


def test_closing_an_owned_client_closes_its_tts_pool(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    with aai.TextToSpeech(api_key="k") as tts:
        tts.synthesize("Hi.")
        pool = tts.client.tts_http_client
    assert pool.is_closed
    assert tts.client.http_client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        tts.synthesize("Hi.")

    # and the shared default client's pool stays open
    shared = aai.TextToSpeech()
    pool = shared.client.tts_http_client
    shared.close()
    assert not pool.is_closed


def test_client_close_closes_both_pools():
    client = aai.Client(api_key="test")
    tts = aai.TextToSpeech(client=client)
    pools = (client.http_client, client.tts_http_client)

    # A passed client stays the caller's to close
    tts.close()
    assert not any(pool.is_closed for pool in pools)

    client.close()
    assert all(pool.is_closed for pool in pools)


def test_a_client_closed_before_it_spoke_fails_as_a_closed_pool_does():
    client = aai.Client(api_key="test")
    client.close()

    # Construction builds no pool on a closed client and does not raise
    tts = aai.TextToSpeech(client=client)
    assert client._tts_http_client is None

    with pytest.raises(RuntimeError, match="closed"):
        tts.synthesize("Hi.")


def test_a_request_within_the_tts_keepalive_expiry_reuses_the_connection():
    # Given: a Text-to-Speech pool with the default 120 s idle time
    with serving(_AUDIO, _OK_HEADERS) as server:
        client = aai.Client(
            settings=aai.Settings(api_key="test", tts_base_url=server.url)
        )
        tts = aai.TextToSpeech(client=client)
        try:
            # When: three requests follow each other
            tts.synthesize("One.")
            tts.synthesize("Two.")
            with tts.stream("Three.") as speech:
                assert speech.read() == _AUDIO

            # Then: they all use the first one's connection
            assert server.connections == 1
        finally:
            client.close()


def test_a_request_after_the_tts_keepalive_expiry_opens_a_new_connection():
    # Given: a Text-to-Speech pool that keeps idle connections 0.2 s
    with serving(_AUDIO, _OK_HEADERS) as server:
        settings = aai.Settings(
            api_key="test", tts_base_url=server.url, tts_keepalive_expiry=0.2
        )
        client = aai.Client(settings=settings)
        tts = aai.TextToSpeech(client=client)
        try:
            tts.synthesize("One.")

            # When: the next request comes after the connection's idle time
            time.sleep(0.5)
            with tts.stream("Two.") as speech:
                assert speech.read() == _AUDIO

            # Then: it opens a new connection
            assert server.connections == 2
        finally:
            client.close()


def test_a_connection_the_server_closed_while_idle_is_not_reused():
    # Given: a server (or load balancer) that closes each connection 0.2 s
    # after its response, well before the pool's 120 s idle time runs out
    with serving(_AUDIO, _OK_HEADERS, close_after=0.2) as server:
        settings = aai.Settings(
            api_key="test", tts_base_url=server.url, tts_keepalive_expiry=120.0
        )
        client = aai.Client(settings=settings)
        tts = aai.TextToSpeech(client=client)
        try:
            tts.synthesize("One.")

            # When: the next request comes after the server closed it
            time.sleep(0.5)
            response = tts.synthesize("Two.")

            # Then: it succeeds on a new connection, with no
            # RemoteProtocolError from sending on the closed one
            assert response.audio == _AUDIO
            assert server.connections == 2
        finally:
            client.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_does_not_share_the_parents_tts_connections():
    with serving(_AUDIO, _OK_HEADERS) as server:
        settings = aai.Settings(api_key="test", tts_base_url=server.url)
        client = aai.Client(settings=settings)
        tts = aai.TextToSpeech(client=client)
        try:
            # Given: a parent whose pool holds an idle connection
            tts.synthesize("Warm-up.")
            parent_pool = client.tts_http_client
            assert server.connections == 1

            # When: a forked child makes a request
            with warnings.catch_warnings():
                # Python 3.12 warns about forking a process that has threads.
                warnings.simplefilter("ignore", DeprecationWarning)
                pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    dropped = client._tts_http_client is None
                    response = tts.synthesize("Child.")
                    fresh = client.tts_http_client is not parent_pool
                    code = 0 if dropped and fresh and response.audio else 2
                finally:
                    os._exit(code)
            _, status = os.waitpid(pid, 0)

            # Then: the child used a pool and a connection of its own
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            assert server.connections == 2
            # and the parent's connection is intact and still reused
            assert client.tts_http_client is parent_pool
            tts.synthesize("Parent.")
            assert server.connections == 2
        finally:
            client.close()


def test_a_forked_child_drops_the_inherited_tts_pool_without_closing_it(
    monkeypatch: pytest.MonkeyPatch,
):
    # Closing a pool takes its lock, which another thread of the parent may
    # have held at the fork, so a child that closes it can wait forever.
    closed: List[str] = []
    close = httpx.Client.close

    def recording_close(self: httpx.Client) -> None:
        closed.append(repr(self))
        close(self)

    with serving(_AUDIO, _OK_HEADERS) as server:
        settings = aai.Settings(api_key="test", tts_base_url=server.url)
        client = aai.Client(settings=settings)
        tts = aai.TextToSpeech(client=client)
        try:
            # Given: a pool that has sent a request and holds an idle
            # connection, as a warm-up before the fork leaves it
            tts.synthesize("Warm-up.")
            inherited = weakref.ref(client.tts_http_client)
            monkeypatch.setattr(httpx.Client, "close", recording_close)

            # When: the fork handler runs in the child, and the pool it
            # dropped is collected
            with warnings.catch_warnings():
                # The collected pool's sockets warn that nothing closed them.
                warnings.simplefilter("ignore", ResourceWarning)
                client_module._forget_tts_pools_in_child()
                gc.collect()

            # Then: nothing closed the pool, and it is gone
            assert closed == []
            assert inherited() is None
        finally:
            monkeypatch.undo()
            client.close()


def test_the_httpx_floor_excludes_releases_that_close_a_collected_client():
    # httpx 0.19 to 0.21 close a client in `Client.__del__`, so on them the
    # pool the fork handler drops would take its lock when it is collected.
    setup_py = (Path(__file__).parents[2] / "setup.py").read_text()
    floor = re.search(r'"httpx>=([0-9.]+)"', setup_py)

    assert floor is not None
    assert tuple(int(part) for part in floor.group(1).split(".")) >= (0, 22)
