"""Tests for the asyncio Diarization API client.

The request shape is covered by `test_diarization.py`; these pin the asyncio
twin: same route, same parts, same parsing, same errors, and its pool.
"""

import json

import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from tests.unit.test_diarization import (
    _OK_RESPONSE,
    _SEGMENTS_ONLY,
    _WORDS,
    DIARIZE_URL,
    _json_part,
    _mock_ok,
    _sent,
)

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"


async def test_diarize_with_a_sync_transcript(httpx_mock: HTTPXMock):
    # GIVEN a sync transcription result
    _mock_ok(httpx_mock)
    transcript = aai.SyncTranscriptResponse(
        text="hello hi",
        words=[aai.SyncWord(**w) for w in _WORDS],
        confidence=0.92,
        audio_duration_ms=9500,
        session_id="sync-session",
    )

    # WHEN it is diarized asynchronously
    async with aai.AsyncDiarizer() as diarizer:
        result = await diarizer.diarize(b"RIFF", transcript=transcript)

    # THEN the same parts were sent and the result parsed
    parts = _sent(httpx_mock)
    assert list(parts) == ["transcript", "audio"]
    assert _json_part(parts["transcript"]) == {"words": _WORDS}
    assert httpx_mock.get_requests()[0].headers["authorization"] == "test"
    assert [u.speaker for u in result.utterances] == ["A", "B"]
    assert result.session_id == _OK_RESPONSE["session_id"]


async def test_diarize_segments_only_with_pcm(httpx_mock: HTTPXMock, tmp_path):
    # GIVEN a raw PCM file
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    path = tmp_path / "call.raw"
    path.write_bytes(b"\x00\x01" * 160)
    config = aai.DiarizationConfig(sample_rate=8000, channels=2)

    # WHEN it is diarized without a transcript
    async with aai.AsyncDiarizer(config=config) as diarizer:
        result = await diarizer.diarize(path)

    # THEN config and PCM audio were sent, and segments came back
    parts = _sent(httpx_mock)
    assert list(parts) == ["config", "audio"]
    assert json.loads(parts["config"].get_payload(decode=True)) == {
        "sample_rate": 8000,
        "channels": 2,
    }
    assert parts["audio"].get_content_type() == "audio/pcm"
    assert result.words == [] and len(result.segments) == 2


async def test_unauthorized_maps_to_diarization_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=DIARIZE_URL,
        method="POST",
        status_code=401,
        json={"status": 401, "title": "Unauthorized", "detail": "Invalid key"},
        headers={"content-type": "application/problem+json"},
    )

    async with aai.AsyncDiarizer() as diarizer:
        with pytest.raises(aai.DiarizationError) as exc_info:
            await diarizer.diarize(b"RIFF")

    assert exc_info.value.status_code == 401
    assert exc_info.value.error_code == "unauthorized"


async def test_custom_base_url_on_a_shared_client(httpx_mock: HTTPXMock):
    # GIVEN a shared client pointed at the US host
    _mock_ok(
        httpx_mock,
        _SEGMENTS_ONLY,
        url="https://diarization.us.assemblyai.com/v1/diarize",
    )
    settings = aai.Settings(
        api_key="us-key", diarization_base_url="https://diarization.us.assemblyai.com"
    )

    async with aai.AsyncClient(settings=settings) as client:
        diarizer = aai.AsyncDiarizer(client=client)
        await diarizer.diarize(b"RIFF")
        # WHEN the diarizer is closed, THEN the shared client stays open
        await diarizer.aclose()
        assert not client.http_client.is_closed

    assert httpx_mock.get_requests()[0].headers["authorization"] == "us-key"


async def test_rejects_other_products_configs():
    with pytest.raises(TypeError, match="expects DiarizationConfig"):
        aai.AsyncDiarizer(config=aai.TTSConfig())

    async with aai.AsyncDiarizer() as diarizer:
        with pytest.raises(TypeError, match="expects DiarizationConfig"):
            await diarizer.diarize(b"RIFF", config=aai.SyncTranscriptionConfig())


async def test_owned_client_is_closed():
    diarizer = aai.AsyncDiarizer()
    await diarizer.aclose()

    assert diarizer.client.http_client.is_closed
