import asyncio
import json
import queue
import threading

import pytest

from assemblyai.streaming.v3 import (
    AsyncRealTimeTranscriber,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    TurnEvent,
)
from assemblyai.streaming.v3._base import _dump_model, _dump_model_json


def turn_payload(scores):
    payload = {
        "type": "Turn",
        "turn_order": 0,
        "turn_is_formatted": False,
        "end_of_turn": True,
        "transcript": "Hello",
        "end_of_turn_confidence": 0.9,
        "speaker_label": "A",
        "words": [
            {
                "start": 0,
                "end": 100,
                "confidence": 0.99,
                "text": "Hello",
                "word_is_final": True,
                "speaker": "A",
            }
        ],
    }
    if scores is not None:
        payload["speaker_confidence"], payload["words"][0]["speaker_confidence"] = (
            scores
        )
    return payload


def assert_turn(event, payload, scores):
    assert isinstance(event, TurnEvent)
    assert event.speaker_confidence == (scores[0] if scores is not None else None)
    assert event.words[0].speaker_confidence == (
        scores[1] if scores is not None else None
    )
    assert _dump_model(event) == payload
    assert json.loads(_dump_model_json(event)) == payload


class SyncSocket:
    def __init__(self, payload):
        self.incoming = queue.Queue()
        self.incoming.put(json.dumps(payload))
        self.closed = False

    def recv(self, timeout=1):
        try:
            return self.incoming.get(timeout=min(timeout, 0.05))
        except queue.Empty:
            raise TimeoutError

    def close(self):
        self.closed = True


@pytest.mark.parametrize("scores", [(0.938865, 0.976504), (0.0, 0.0), None])
def test_sync_callback_preserves_speaker_confidence(mocker, scores):
    payload = turn_payload(scores)
    socket = SyncSocket(payload)
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect", return_value=socket
    )
    client = RealTimeTranscriber(api_key="offline-test")
    received = []
    ready = threading.Event()

    def on_turn(c, event):
        received.append((c, event))
        ready.set()

    client.on(RealTimeEvents.Turn, on_turn)
    try:
        client.connect(
            RealTimeParameters(sample_rate=16000, speech_model="universal-3-5-pro")
        )
        assert ready.wait(2)
        assert received[0][0] is client
        assert_turn(received[0][1], payload, scores)
    finally:
        client.disconnect()
        client._client._http_client.close()
    assert socket.closed


class AsyncSocket:
    def __init__(self, payload):
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait(json.dumps(payload))
        self.closed = False

    async def recv(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("scores", [(0.938865, 0.976504), (0.0, 0.0), None])
async def test_async_callback_preserves_speaker_confidence(mocker, scores):
    payload = turn_payload(scores)
    socket = AsyncSocket(payload)

    async def connect(*args, **kwargs):
        return socket

    mocker.patch(
        "assemblyai.streaming.v3.async_client.websocket_connect_async", new=connect
    )
    client = AsyncRealTimeTranscriber(api_key="offline-test")
    received = []
    ready = asyncio.Event()

    async def on_turn(c, event):
        received.append((c, event))
        ready.set()

    client.on(RealTimeEvents.Turn, on_turn)
    try:
        await client.connect(
            RealTimeParameters(sample_rate=16000, speech_model="universal-3-5-pro")
        )
        await asyncio.wait_for(ready.wait(), timeout=2)
        assert received[0][0] is client
        assert_turn(received[0][1], payload, scores)
    finally:
        await client.disconnect()
    assert socket.closed
