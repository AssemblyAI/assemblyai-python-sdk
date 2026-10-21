"""Tests for the asyncio Text-to-Speech client.

The body shape and the error parsing are covered by `test_tts.py`; these pin
the asyncio twin's behaviour: same route, same audio and headers, same errors,
plus pool ownership.
"""

import json
from typing import AsyncIterator, Optional

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

SPEECH_URL = f"{aai.settings.tts_base_url}/v1/speech"

_AUDIO = b"\x00\x01" * 480

_OK_HEADERS = {
    "content-type": "audio/pcm;rate=24000;channels=1",
    "x-request-id": "req-async",
    "x-tts-model": "assemblyai-tts-1",
    "x-tts-voice-id": "aai-alba",
    "x-tts-language": "english",
    "x-tts-encoding": "pcm_s16le",
    "x-tts-sample-rate": "24000",
    "x-tts-container": "raw",
    "x-tts-billed-characters": "12",
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


class _RecordingStream(httpx.AsyncByteStream):
    """A response body that yields `chunks`, then raises `error` if one is
    given, and records whether it was closed."""

    def __init__(self, *chunks: bytes, error: Optional[Exception] = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True


async def test_synthesize_matches_the_threaded_twin(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock, **{"x-tts-audio-duration-ms": "20"})

    async with aai.AsyncTextToSpeech(config=aai.TTSConfig(voice_id="aai-alba")) as tts:
        response = await tts.synthesize("Hello there.")

    request = httpx_mock.get_requests()[0]
    assert str(request.url) == SPEECH_URL
    assert request.headers["authorization"] == "test"
    assert _sent_body(httpx_mock) == {
        "text": "Hello there.",
        "voice_id": "aai-alba",
        "stream": False,
    }
    assert response.audio == _AUDIO
    assert response.request_id == "req-async"
    assert response.voice_id == "aai-alba"
    assert response.sample_rate == 24000
    assert response.audio_duration_ms == 20


async def test_stream_yields_audio_and_reports_the_headers(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    async with aai.AsyncTextToSpeech() as tts:
        async with await tts.stream("Hello there.") as speech:
            chunks = [chunk async for chunk in speech]

    assert _sent_body(httpx_mock) == {"text": "Hello there.", "stream": True}
    assert b"".join(chunks) == _AUDIO
    assert speech.info.model == "assemblyai-tts-1"
    assert speech.info.voice_id == "aai-alba"


async def test_stream_aread_collects_the_whole_audio(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    async with aai.AsyncTextToSpeech() as tts:
        speech = await tts.stream("Hello there.")
        try:
            assert await speech.aread() == _AUDIO
        finally:
            await speech.aclose()


async def test_errors_are_tts_errors(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.UNPROCESSABLE_ENTITY,
        json={
            "code": "voice_not_found",
            "message": "Voice not found",
            "param": "voice_id",
            "request_id": "req-2",
            "errors": None,
            "retryable": False,
            "retry_after_seconds": None,
        },
    )
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"code": "at_capacity", "message": "No room", "retryable": True},
        headers={"retry-after": "3"},
    )

    async with aai.AsyncTextToSpeech() as tts:
        with pytest.raises(aai.TTSError) as exc_info:
            await tts.synthesize("Hi.", aai.TTSConfig(voice_id="aai-nobody"))
        assert exc_info.value.error_code == "voice_not_found"
        assert exc_info.value.param == "voice_id"
        assert exc_info.value.request_id == "req-2"

        with pytest.raises(aai.TTSError) as exc_info:
            await tts.stream("Hi.")
        assert exc_info.value.error_code == "at_capacity"
        assert exc_info.value.retry_after == 3


async def test_a_shared_client_is_left_open_on_aclose(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)

    client = aai.AsyncClient(api_key="test")
    async with aai.AsyncTextToSpeech(client=client) as tts:
        await tts.synthesize("One.")

    # The pool is still usable after the instance closed
    await aai.AsyncTextToSpeech(client=client).synthesize("Two.")
    await client.aclose()

    assert len(httpx_mock.get_requests()) == 2


async def test_wrong_config_type_is_a_type_error():
    with pytest.raises(TypeError):
        aai.AsyncTextToSpeech(config=aai.DictationConfig())  # type: ignore[arg-type]


async def test_biometric_terms_refusal_and_auth_unavailable(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.FORBIDDEN,
        json={
            "code": "biometric_terms_not_accepted",
            "message": "This account has not accepted the biometric data terms.",
            "param": "extend_biometric_consent",
            "request_id": "req-403",
            "errors": None,
            "retryable": False,
            "retry_after_seconds": None,
        },
    )
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"code": "auth_unavailable", "message": "Retry", "retryable": True},
        headers={"retry-after": "1"},
    )
    renew = aai.TTSConfig(voice_id="my-voice", extend_biometric_consent=True)

    async with aai.AsyncTextToSpeech() as tts:
        with pytest.raises(aai.TTSError) as exc_info:
            await tts.synthesize("Hi.", renew)
        assert exc_info.value.status_code == 403
        assert exc_info.value.error_code == "biometric_terms_not_accepted"
        assert exc_info.value.retryable is False

        with pytest.raises(aai.TTSError) as exc_info:
            await tts.stream("Hi.", renew)
        assert exc_info.value.error_code == "auth_unavailable"
        assert exc_info.value.retryable is True
        assert exc_info.value.retry_after == 1


async def test_a_stream_cut_after_the_first_audio_is_a_retryable_tts_error(
    httpx_mock: HTTPXMock,
):
    # GIVEN a streamed answer that the API cuts after its first audio
    body = _RecordingStream(
        _AUDIO,
        error=httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        ),
    )
    httpx_mock.add_callback(
        lambda request: httpx.Response(200, headers=_OK_HEADERS, stream=body),
        url=SPEECH_URL,
        method="POST",
    )

    received = []
    async with aai.AsyncTextToSpeech() as tts:
        with pytest.raises(aai.TTSError) as exc_info:
            async with await tts.stream("Hello there.") as speech:
                async for chunk in speech:
                    received.append(chunk)

    # THEN the first audio arrived, then a retryable error with the request id
    assert b"".join(received) == _AUDIO
    err = exc_info.value
    assert err.status_code == 200
    assert err.retryable is True
    assert err.request_id == "req-async"
    assert isinstance(err.__cause__, httpx.RemoteProtocolError)
    # AND the connection was released
    assert body.closed


async def test_stream_aread_raises_the_same_error_on_a_cut(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(
        lambda request: httpx.Response(
            200,
            headers=_OK_HEADERS,
            stream=_RecordingStream(_AUDIO, error=httpx.ReadTimeout("no audio")),
        ),
        url=SPEECH_URL,
        method="POST",
    )

    async with aai.AsyncTextToSpeech() as tts:
        async with await tts.stream("Hello there.") as speech:
            with pytest.raises(aai.TTSError) as exc_info:
                await speech.aread()

    assert exc_info.value.retryable is True
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


async def test_synthesize_and_stream_use_their_own_timeouts(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    client = aai.AsyncClient(api_key="test")
    client.settings.tts_http_timeout = 7.0
    client.settings.tts_synthesize_http_timeout = 99.0

    async with client:
        tts = aai.AsyncTextToSpeech(client=client)
        await tts.synthesize("Hi.")
        async with await tts.stream("Hi.") as speech:
            await speech.aread()

    synthesize, stream = [r.extensions["timeout"] for r in httpx_mock.get_requests()]
    assert synthesize == {"connect": 7.0, "read": 99.0, "write": 7.0, "pool": 7.0}
    assert stream == {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}


async def test_an_owned_pool_is_closed_on_aclose():
    tts = aai.AsyncTextToSpeech(api_key="test")

    await tts.aclose()

    assert tts.client.http_client.is_closed


async def test_voice_is_refused_by_the_config():
    with pytest.raises(ValueError, match="voice_id"):
        aai.AsyncTextToSpeech(config=aai.TTSConfig(voice="alba"))  # type: ignore[call-arg]
