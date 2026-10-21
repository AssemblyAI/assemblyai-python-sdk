"""Tests for the asyncio Text-to-Speech client.

The body shape and the error parsing are covered by `test_tts.py`; these pin
the asyncio twin's behaviour: same route, same audio and headers, same errors,
plus pool ownership.
"""

import json

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
    "x-tts-voice": "jane",
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

    async with aai.AsyncTextToSpeech(config=aai.TTSConfig(voice="alba")) as tts:
        response = await tts.synthesize("Hello there.")

    request = httpx_mock.get_requests()[0]
    assert str(request.url) == SPEECH_URL
    assert request.headers["authorization"] == "test"
    assert _sent_body(httpx_mock) == {
        "text": "Hello there.",
        "voice": "alba",
        "stream": False,
    }
    assert response.audio == _AUDIO
    assert response.request_id == "req-async"
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
            await tts.synthesize("Hi.", aai.TTSConfig(voice_id="nope"))
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
