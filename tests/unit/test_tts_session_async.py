"""Tests for the asyncio Text-to-Speech WebSocket session."""

import asyncio
import base64
import json
from typing import Any, List, Optional
from urllib.parse import parse_qs, urlsplit

import pytest
import pytest_asyncio
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

import assemblyai as aai

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

_BEGIN = {
    "type": "Begin",
    "id": "sess-1",
    "expires_at": 1789556741,
    "configuration": {"model": "assemblyai-tts-1"},
}
_TERMINATION = {
    "type": "Termination",
    "session_duration_seconds": 3,
    "total_input_char_length": 20,
    "audio_duration_seconds": 1.5,
}


class _FakeAsyncServer:
    def __init__(self, begin: Optional[dict] = _BEGIN, answer_flush: bool = True):
        self.sent: List[dict] = []
        self.inbox: "asyncio.Queue[Any]" = asyncio.Queue()
        self.closed = False
        self.text = ""
        self.flush_id = 0
        self.answer_flush = answer_flush
        if begin is not None:
            self.push(begin)

    def push(self, item: Any) -> None:
        self.inbox.put_nowait(json.dumps(item) if isinstance(item, dict) else item)

    async def send(self, raw: str) -> None:
        frame = json.loads(raw)
        self.sent.append(frame)
        kind = frame["type"]
        if kind == "Generate":
            self.text += frame["text"]
        elif kind == "Flush":
            if self.answer_flush:
                audio = base64.b64encode(self.text.encode()).decode()
                self.push({"type": "Audio", "audio": audio, "flush_id": self.flush_id})
                self.push(
                    {
                        "type": "FlushDone",
                        "flush_id": self.flush_id,
                        "audio_duration_ms": 80,
                    }
                )
            self.flush_id += 1
            self.text = ""
        elif kind == "Cancel":
            self.flush_id += 1
            self.text = ""
            self.push({"type": "Cancelled", "audio_duration_seconds": 0.1})
        elif kind == "Terminate":
            self.push(_TERMINATION)
            self.push(ConnectionClosedOK(Close(1000, "Session ended"), None))

    async def recv(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True
        self.push(ConnectionClosedOK(Close(1000, ""), None))


@pytest_asyncio.fixture
async def server(mocker):
    # Built inside the running loop: on Python 3.9 an asyncio.Queue binds to
    # the loop that exists when it is created.
    fake = _FakeAsyncServer()

    async def connect(uri, additional_headers):
        connect.uri = uri
        connect.headers = additional_headers
        return fake

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )
    fake.connect = connect
    return fake


async def test_async_with_connects_speaks_and_terminates(server):
    async with aai.AsyncTTSSession(
        config=aai.TTSConfig(voice_id="aai-alba")
    ) as session:
        assert session.begin is not None and session.begin.id == "sess-1"
        chunks = [chunk async for chunk in session.speak("Hello there.")]

    assert b"".join(chunks) == b"Hello there."
    assert parse_qs(urlsplit(server.connect.uri).query)["voice_id"] == ["aai-alba"]
    assert server.connect.headers["Authorization"] == "test"
    assert [f["type"] for f in server.sent] == ["Generate", "Flush", "Terminate"]
    assert session.termination is not None
    assert server.closed


async def test_pipelined_requests_and_cancel(server):
    async with aai.AsyncTTSSession() as session:
        await session.generate("One. ")
        first = await session.flush()
        second = await session.flush()
        assert b"".join([c async for c in session.audio(first)]) == b"One. "
        assert b"".join([c async for c in session.audio(second)]) == b""

        server.answer_flush = False
        third = await session.flush()
        await session.cancel()
        assert [c async for c in session.audio(third)] == []


async def test_an_error_before_begin_raises_from_async_with(mocker):
    fake = _FakeAsyncServer(begin=None)
    fake.push(
        {
            "type": "Error",
            "error_code": 3009,
            "error": "Too many sessions",
            "error_type": "rate_limited",
            "retryable": True,
            "param": None,
            "retry_after_seconds": 17,
        }
    )

    async def connect(uri, additional_headers):
        return fake

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )

    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession():
            pass
    assert exc_info.value.error_code == "rate_limited"
    assert exc_info.value.retry_after == 17
    assert fake.closed


async def test_handlers_may_be_coroutines(server):
    received = []

    async def on_audio(event):
        received.append(event.audio)

    async with aai.AsyncTTSSession() as session:
        session.on(aai.TTSEvents.Audio, on_audio)
        await session.generate("Pushed.")
        flush_id = await session.flush()
        assert [c async for c in session.audio(flush_id)] == []

    assert received == [b"Pushed."]


async def test_a_lost_connection_raises_a_retryable_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(ConnectionClosedError(None, None))  # abnormal: no close frame
            [c async for c in session.audio(flush_id)]
    assert exc_info.value.retryable is True
    assert exc_info.value.body["close_code"] == 1006


async def test_a_cancel_drops_every_pipelined_request(server):
    server.answer_flush = False
    async with aai.AsyncTTSSession() as session:
        first = await session.flush()
        second = await session.flush()
        await session.cancel()
        assert [c async for c in session.audio(first)] == []
        assert [c async for c in session.audio(second)] == []


async def test_a_session_object_can_be_reused(mocker):
    fakes = [_FakeAsyncServer(), _FakeAsyncServer()]

    async def connect(uri, additional_headers):
        return fakes.pop(0)

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )
    session = aai.AsyncTTSSession()
    async with session:
        assert b"".join([c async for c in session.speak("One.")]) == b"One."
    async with session:
        assert b"".join([c async for c in session.speak("Two.")]) == b"Two."


async def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setattr(aai.settings, "api_key", None)
    with pytest.raises(ValueError, match="api_key= or token="):
        aai.AsyncTTSSession()
