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
from assemblyai.tts.v1 import async_api

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

SPEECH_URL = f"{aai.settings.tts_base_url}/v1/speech"

_AUDIO = b"\x00\x01" * 480

_OK_HEADERS = {
    "content-type": "audio/pcm;rate=24000;channels=1",
    "x-request-id": "req-async",
    "x-tts-region": "us-east-1",
    "x-tts-model": "assemblyai-tts-1",
    "x-tts-voice-id": "aai-jane",
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
    assert response.region == "us-east-1"
    assert response.voice_id == "aai-jane"
    assert response.sample_rate == 24000
    assert response.audio_duration_ms == 20


async def test_stream_yields_audio_and_reports_the_headers(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    async with aai.AsyncTextToSpeech() as tts:
        speech = tts.stream("Hello there.")
        assert httpx_mock.get_requests() == []
        async with speech:
            chunks = [chunk async for chunk in speech]

    assert _sent_body(httpx_mock) == {"text": "Hello there.", "stream": True}
    assert b"".join(chunks) == _AUDIO
    assert speech.info is not None and speech.info.model == "assemblyai-tts-1"


async def test_stream_aread_collects_the_whole_audio(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)

    async with aai.AsyncTextToSpeech() as tts:
        speech = tts.stream("Hello there.")
        try:
            assert await speech.aread() == _AUDIO
        finally:
            await speech.aclose()


class _CutStream(httpx.AsyncByteStream):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
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
async def test_a_body_cut_after_the_first_audio_is_a_retryable_tts_error(error):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers=_OK_HEADERS, stream=_CutStream(error)
        )
    )
    async with httpx.AsyncClient(transport=transport) as http:
        speech = async_api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        async with speech:
            with pytest.raises(aai.TTSError) as exc_info:
                await speech.aread()

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 200
    assert exc_info.value.request_id == "req-async"
    assert exc_info.value.__cause__ is error


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
    httpx_mock.add_response(
        url=SPEECH_URL,
        method="POST",
        status_code=httpx.codes.FORBIDDEN,
        json={"code": "biometric_terms_not_accepted", "message": "Accept the terms"},
    )

    async with aai.AsyncTextToSpeech() as tts:
        with pytest.raises(aai.TTSError) as exc_info:
            await tts.synthesize("Hi.", aai.TTSConfig(voice_id="nope"))
        assert exc_info.value.error_code == "voice_not_found"
        assert exc_info.value.param == "voice_id"
        assert exc_info.value.request_id == "req-2"

        with pytest.raises(aai.TTSError) as exc_info:
            async with tts.stream("Hi."):
                pass
        assert exc_info.value.error_code == "at_capacity"
        assert exc_info.value.retry_after == 3

        with pytest.raises(aai.TTSError) as exc_info:
            await tts.synthesize("Hi.", aai.TTSConfig(voice_id="aai-x"))
        assert exc_info.value.status_code == 403
        assert exc_info.value.error_code == "biometric_terms_not_accepted"
        assert exc_info.value.retryable is False


async def test_synthesize_and_stream_use_their_own_timeouts(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    client = aai.AsyncClient(api_key="test")
    client.settings.tts_http_timeout = 7.0
    client.settings.tts_synthesize_http_timeout = 99.0

    async with aai.AsyncTextToSpeech(client=client) as tts:
        await tts.synthesize("Hi.")
        async with tts.stream("Hi.") as speech:
            await speech.aread()
    await client.aclose()

    timeouts = [r.extensions["timeout"] for r in httpx_mock.get_requests()]
    assert (timeouts[0]["read"], timeouts[0]["connect"]) == (99.0, 7.0)
    assert timeouts[1]["read"] == 7.0


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


async def test_a_refused_stream_releases_its_connection_if_its_body_fails(
    httpx_mock: HTTPXMock,
):
    # GIVEN a refusal whose error body is cut while it is read
    body = _RecordingStream(b'{"code": ', error=httpx.ReadError("reset"))
    httpx_mock.add_callback(
        lambda request: httpx.Response(
            503, headers={"content-type": "application/json"}, stream=body
        ),
        url=SPEECH_URL,
        method="POST",
    )

    # THEN the network error surfaces and the connection is released
    async with aai.AsyncTextToSpeech() as tts:
        with pytest.raises(httpx.ReadError):
            async with tts.stream("Hi."):
                pass

        assert body.closed


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
