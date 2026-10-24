"""Tests for the asyncio Text-to-Speech client.

The body shape and the error parsing are covered by `test_tts.py`; these pin
the asyncio twin's behaviour: same route, same audio and headers, same errors,
plus pool ownership.
"""

import json
from typing import AsyncIterator

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
    assert speech.info.model == "assemblyai-tts-1"


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


class _RecordingStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _AUDIO[:100]
        yield _AUDIO[100:]

    async def aclose(self) -> None:
        self.closed = True


def _transport(body: httpx.AsyncByteStream, status: int = 200) -> httpx.MockTransport:
    headers = _OK_HEADERS if status == 200 else {"content-type": "application/json"}
    return httpx.MockTransport(
        lambda request: httpx.Response(status, headers=headers, stream=body)
    )


async def test_leaving_the_async_with_block_early_releases_the_connection():
    body = _RecordingStream()
    async with httpx.AsyncClient(transport=_transport(body)) as http:
        async with async_api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        ) as speech:
            await speech.aiter_bytes().__anext__()  # a barge-in after the first chunk
        assert body.closed


class _JsonBody(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield json.dumps({"code": "at_capacity", "message": "No room"}).encode()

    async def aclose(self) -> None:
        self.closed = True


async def test_a_refused_stream_releases_its_connection():
    body = _JsonBody()
    async with httpx.AsyncClient(transport=_transport(body, status=503)) as http:
        speech = async_api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        with pytest.raises(aai.TTSError) as exc_info:
            await speech.aread()
    assert exc_info.value.error_code == "at_capacity"
    assert body.closed


class _CutJsonBody(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"code": '
        raise httpx.ReadError("connection reset")

    async def aclose(self) -> None:
        self.closed = True


async def test_a_refused_stream_releases_its_connection_if_its_body_fails():
    # httpx closes a fully read body itself; a cut body is released only by
    # the stream. Holding the error keeps `cm` alive, so GC cannot do it.
    body = _CutJsonBody()
    async with httpx.AsyncClient(transport=_transport(body, status=503)) as http:
        speech = async_api.stream_speech(
            http, base_url="https://tts.test", body={"text": "Hi."}, timeout=5.0
        )
        with pytest.raises(httpx.ReadError) as exc_info:
            await speech.aread()
        assert body.closed
        assert "connection reset" in str(exc_info.value)


async def test_an_owned_pool_is_closed_on_aclose():
    tts = aai.AsyncTextToSpeech(api_key="k")
    http = tts.client.http_client
    await tts.aclose()
    assert http.is_closed


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


async def test_the_tts_pool_keeps_idle_connections_120_s():
    client = aai.AsyncClient(settings=aai.Settings(api_key="test"))
    pool = client.tts_http_client

    # A pool of its own, speaking HTTP/1.1, holding idle connections 120 s
    assert pool is not client.http_client
    assert pool is client.tts_http_client
    assert pool._transport._pool._keepalive_expiry == 120.0  # type: ignore[attr-defined]
    assert pool._transport._pool._http2 is False  # type: ignore[attr-defined]
    # while the pool of the other APIs keeps httpx's default
    assert client.http_client._transport._pool._keepalive_expiry == 5.0  # type: ignore[attr-defined]
    await client.aclose()


async def test_requests_go_through_the_tts_pool(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    client = aai.AsyncClient(api_key="test")
    assert client._tts_http_client is None  # built on first use only
    # The pool of the other APIs is not needed
    await client.http_client.aclose()

    async with aai.AsyncTextToSpeech(client=client) as tts:
        await tts.synthesize("Hi.")
        async with tts.stream("Hi.") as speech:
            await speech.aread()

    assert len(httpx_mock.get_requests()) == 2
    assert client.last_response is not None
    assert str(client.last_response.request.url) == SPEECH_URL
    await client.aclose()


async def test_aclose_closes_both_pools(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock)
    tts = aai.AsyncTextToSpeech(api_key="k")
    await tts.synthesize("Hi.")
    pools = (tts.client.http_client, tts.client.tts_http_client)

    await tts.aclose()

    assert all(pool.is_closed for pool in pools)


async def test_a_client_closed_before_it_spoke_fails_as_a_closed_pool_does():
    tts = aai.AsyncTextToSpeech(api_key="k")
    await tts.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await tts.synthesize("Hi.")
