"""Polling delays must not exceed the remaining timeout budget."""

from types import SimpleNamespace

import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.prerecorded.v2 import async_transcript, transcript
from tests.unit import factories


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    async def async_sleep(self, seconds):
        self.sleep(seconds)


def response_and_settings():
    response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()
    settings = aai.Settings(api_key="test", polling_interval=2.0)
    return aai.types.TranscriptResponse(**response), settings


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_poll_sleep_respects_remaining_timeout(httpx_mock: HTTPXMock, monkeypatch):
    response, settings = response_and_settings()
    httpx_mock.add_response(
        url=f"{settings.base_url}/v2/transcript/{response.id}", json=response.dict()
    )
    clock = Clock()
    monkeypatch.setattr(transcript, "time", clock)
    result = aai.Transcript.from_response(
        client=aai.Client(settings=settings), response=response
    )

    with pytest.raises(aai.TranscriptError, match="did not finish"):
        result.wait_for_completion(poll_timeout=0.1)

    assert clock.now == pytest.approx(0.1)


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_async_poll_sleep_respects_remaining_timeout(
    httpx_mock: HTTPXMock, monkeypatch
):
    response, settings = response_and_settings()
    httpx_mock.add_response(
        url=f"{settings.base_url}/v2/transcript/{response.id}", json=response.dict()
    )
    clock = Clock()
    monkeypatch.setattr(async_transcript, "time", clock)
    monkeypatch.setattr(
        async_transcript, "asyncio", SimpleNamespace(sleep=clock.async_sleep)
    )
    async with aai.AsyncClient(settings=settings) as client:
        result = aai.AsyncTranscript.from_response(client=client, response=response)
        with pytest.raises(aai.TranscriptError, match="did not finish"):
            await result.wait_for_completion(poll_timeout=0.1)

    assert clock.now == pytest.approx(0.1)
