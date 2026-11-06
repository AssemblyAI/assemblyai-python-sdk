"""Signed redacted-audio downloads may redirect to a different host."""

import pytest

import assemblyai as aai
from tests.unit import factories


def prepare(httpx_mock, status):
    data = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    data.update(
        redact_pii=True, redact_pii_audio=True, redact_pii_policies=["person_name"]
    )
    response = aai.types.TranscriptResponse(**data)
    settings = aai.Settings(api_key="test")
    httpx_mock.add_response(
        url=f"{settings.base_url}/v2/transcript/{response.id}/redacted-audio",
        json={
            "status": "redacted_audio_ready",
            "redacted_audio_url": "https://audio.example/redacted.wav",
        },
    )
    httpx_mock.add_response(
        url="https://audio.example/redacted.wav",
        status_code=status,
        headers={"location": "https://cdn.example/redacted.wav"},
    )
    httpx_mock.add_response(
        url="https://cdn.example/redacted.wav", content=b"redacted audio"
    )
    return response, settings


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_save_redacted_audio_follows_redirect(httpx_mock, tmp_path, status):
    response, settings = prepare(httpx_mock, status)
    transcript = aai.Transcript.from_response(
        client=aai.Client(settings=settings), response=response
    )
    target = tmp_path / "redacted.wav"
    transcript.save_redacted_audio(str(target))
    assert target.read_bytes() == b"redacted audio"
    assert all(
        "authorization" not in request.headers
        for request in httpx_mock.get_requests()[1:]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_async_save_redacted_audio_follows_redirect(httpx_mock, tmp_path, status):
    response, settings = prepare(httpx_mock, status)
    async with aai.AsyncClient(settings=settings) as client:
        transcript = aai.AsyncTranscript.from_response(client=client, response=response)
        target = tmp_path / "redacted.wav"
        await transcript.save_redacted_audio(str(target))
    assert target.read_bytes() == b"redacted audio"
    assert all(
        "authorization" not in request.headers
        for request in httpx_mock.get_requests()[1:]
    )
