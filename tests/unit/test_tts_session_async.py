"""Tests for the asyncio Text-to-Speech WebSocket session."""

import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, AsyncIterable, List, Optional
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


class _Rejected(Exception):
    """A handshake answered with an HTTP status, as the current clients raise."""

    def __init__(self, status: int) -> None:
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class _LegacyRejected(Exception):
    """The same, as the legacy client of websockets 11 and 12 raises it."""

    def __init__(self, status: int) -> None:
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.status_code = status


_HANDSHAKE_FAILURES = [
    pytest.param(_Rejected(503), 503, True, id="503"),
    pytest.param(_Rejected(502), 502, True, id="502"),
    pytest.param(_Rejected(401), 401, False, id="401"),
    pytest.param(_LegacyRejected(429), 429, True, id="legacy-429"),
    pytest.param(_LegacyRejected(403), 403, False, id="legacy-403"),
    pytest.param(
        ConnectionRefusedError(111, "Connection refused"), None, True, id="refused"
    ),
    pytest.param(TimeoutError("timed out"), None, True, id="timeout"),
    pytest.param(asyncio.TimeoutError(), None, True, id="asyncio-timeout"),
    pytest.param(ValueError("not a WebSocket URI"), None, False, id="bad-uri"),
]


_INACTIVITY_TIMEOUT = {
    "type": "Error",
    "error_code": 3006,
    "error": "No message received for 120 seconds; the idle session was closed.",
    "error_type": "inactivity_timeout",
    "retryable": True,
    "param": None,
    "retry_after_seconds": None,
}
_INPUT_RATE_EXCEEDED = {
    "type": "Error",
    "error_code": 3010,
    "error": "Text arrived faster than it can be spoken.",
    "error_type": "input_rate_exceeded",
    "retryable": False,
    "param": None,
    "retry_after_seconds": None,
}


class _FakeAsyncServer:
    def __init__(self, begin: Optional[dict] = _BEGIN, answer_flush: bool = True):
        self.sent: List[dict] = []
        self.inbox: "asyncio.Queue[Any]" = asyncio.Queue()
        self.closed = False
        self.text = ""
        self.flush_id = 0
        self.answer_flush = answer_flush
        self.answer_terminate = True
        self.closing: Optional[List[Any]] = None
        if begin is not None:
            self.push(begin)

    def close_on_next_send(self, code: int, error: Optional[dict] = None) -> None:
        """The API closes the session: the next send raises, and the reader
        has yet to take in the frames that came first."""
        self.closing = ([error] if error else []) + [
            ConnectionClosedError(Close(code, "See Error message for details"), None)
        ]

    def push(self, item: Any) -> None:
        self.inbox.put_nowait(json.dumps(item) if isinstance(item, dict) else item)

    async def send(self, raw: str) -> None:
        if self.closing is not None:
            closing, self.closing = self.closing, None
            for item in closing:
                self.push(item)
            raise ConnectionClosedError(closing[-1].rcvd, None)
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
        elif kind == "Terminate" and self.answer_terminate:
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


async def _drain(items: AsyncIterable[Any], timeout: float = 5.0) -> List[Any]:
    """Collects `items`, failing the test if it does not end in time."""

    async def collect() -> List[Any]:
        return [item async for item in items]

    return await asyncio.wait_for(collect(), timeout)


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
        chunks = await _drain(session.speak("Hello there."))

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
        assert b"".join(await _drain(session.audio(first))) == b"One. "
        assert b"".join(await _drain(session.audio(second))) == b""

        server.answer_flush = False
        third = await session.flush()
        await session.cancel()
        assert await _drain(session.audio(third)) == []


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
    assert exc_info.value.body["close_code"] == 3009
    assert fake.closed


@pytest.mark.parametrize("failure, status, retryable", _HANDSHAKE_FAILURES)
async def test_a_failed_handshake_keeps_its_status_and_says_if_retryable(
    mocker, failure, status, retryable
):
    async def connect(uri, additional_headers):
        raise failure

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )
    with pytest.raises(aai.TTSError, match="handshake failed") as exc_info:
        async with aai.AsyncTTSSession():
            pass
    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable
    assert exc_info.value.__cause__ is failure


async def test_handlers_may_be_coroutines(server):
    received = []

    async def on_audio(event):
        received.append(event.audio)

    async with aai.AsyncTTSSession() as session:
        session.on(aai.TTSEvents.Audio, on_audio)
        await session.generate("Pushed.")
        flush_id = await session.flush()
        with pytest.raises(RuntimeError, match="Audio handler"):
            await session.audio(flush_id).__anext__()
        await asyncio.sleep(0.05)

    assert received == [b"Pushed."]


async def test_the_error_reaches_waiters_before_the_error_handler(server):
    app_saw_error = asyncio.Event()
    handler_waited: List[bool] = []

    async def on_error(error: aai.TTSError) -> None:
        try:  # waits on the app
            await asyncio.wait_for(app_saw_error.wait(), 5)
            handler_waited.append(True)
        except asyncio.TimeoutError:
            handler_waited.append(False)

    with pytest.raises(aai.TTSError):
        async with aai.AsyncTTSSession() as session:
            session.on(aai.TTSEvents.Error, on_error)
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(_INPUT_RATE_EXCEEDED)
            try:
                await _drain(session.audio(flush_id), timeout=3)
            finally:
                app_saw_error.set()

    assert handler_waited == [True]


async def test_waiting_from_a_handler_raises_instead_of_deadlocking(server):
    refused: List[str] = []

    async def speak_next(event: aai.TTSFlushDoneEvent) -> None:
        try:
            [chunk async for chunk in session.speak("Next.")]
        except RuntimeError as exc:
            refused.append(str(exc))
        try:
            await session.terminate()
        except RuntimeError as exc:
            refused.append(str(exc))

    async with aai.AsyncTTSSession(terminate_timeout=3) as session:
        session.on(aai.TTSEvents.FlushDone, speak_next)
        assert b"".join(await _drain(session.speak("Hi."))) == b"Hi."

    assert len(refused) == 2
    assert all("from a session handler" in message for message in refused)
    assert [f for f in server.sent if f["type"] == "Flush"] == [{"type": "Flush"}]


async def test_a_lost_connection_raises_a_retryable_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(ConnectionClosedError(None, None))  # abnormal: no close frame
            await _drain(session.audio(flush_id))
    assert exc_info.value.retryable is True
    assert exc_info.value.body["close_code"] == 1006


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize(
    "code, retryable",
    [
        (1011, True),
        (1012, True),
        (3005, True),
        (3008, True),
        (3009, True),
        (3006, False),
        (3010, False),
    ],
)
async def test_a_close_without_an_error_frame_is_retryable_by_its_code(
    server, code, retryable
):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(ConnectionClosedError(Close(code, "Upstream failed"), None))
            await _drain(session.audio(flush_id))

    assert exc_info.value.retryable is retryable
    assert exc_info.value.body == {"close_code": code, "reason": "Upstream failed"}


async def test_a_socket_error_is_a_retryable_lost_connection(server):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(OSError("Connection reset by peer"))
            await _drain(session.audio(flush_id))

    assert exc_info.value.retryable is True
    assert exc_info.value.body == {"close_code": 1006, "reason": ""}


async def test_a_failure_in_the_reader_still_ends_the_iteration(server, mocker):
    mocker.patch(
        "assemblyai.tts.v1.async_session.error_from_close",
        side_effect=RuntimeError("bug"),
    )
    with pytest.raises(aai.TTSError, match="reader failed"):
        async with aai.AsyncTTSSession() as session:
            server.answer_flush = False
            flush_id = await session.flush()
            server.push(ConnectionClosedError(None, None))
            await _drain(session.audio(flush_id))


async def test_a_send_the_api_closed_raises_the_sessions_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.close_on_next_send(3010, _INPUT_RATE_EXCEEDED)
            await session.generate("Too much, too fast.")

    assert exc_info.value.error_code == "input_rate_exceeded"
    assert exc_info.value.retryable is False
    assert exc_info.value is session.error


async def test_a_send_on_a_bare_close_raises_a_tts_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        async with aai.AsyncTTSSession() as session:
            server.close_on_next_send(1012)
            await session.flush()

    assert exc_info.value.retryable is True
    assert exc_info.value.body["close_code"] == 1012


async def test_a_cancel_drops_every_pipelined_request(server):
    server.answer_flush = False
    async with aai.AsyncTTSSession() as session:
        first = await session.flush()
        second = await session.flush()
        await session.cancel()
        assert await _drain(session.audio(first)) == []
        assert await _drain(session.audio(second)) == []


async def test_an_overflow_drops_audio_but_never_the_end_of_a_request(
    server, mocker, caplog
):
    mocker.patch("assemblyai.tts.v1.async_session.PULL_QUEUE_FRAMES", 6)
    server.answer_flush = False
    done: List[int] = []
    async with aai.AsyncTTSSession() as session:
        session.on(aai.TTSEvents.FlushDone, lambda event: done.append(event.flush_id))
        flush_ids = [await session.flush() for _ in range(3)]
        for flush_id in flush_ids:  # 4 frames each, all before anything is read
            for _ in range(4):
                server.push({"type": "Audio", "audio": "AA==", "flush_id": flush_id})
            server.push(
                {"type": "FlushDone", "flush_id": flush_id, "audio_duration_ms": 320}
            )
        for _ in range(500):
            if len(done) == 3:
                break
            await asyncio.sleep(0.01)

        got = [await _drain(session.audio(flush_id)) for flush_id in flush_ids]

    # 12 frames for room for 6: the oldest 6 went, every request still ended
    assert [len(chunks) for chunks in got] == [0, 2, 4]
    assert "oldest are being dropped" in caplog.text


async def test_a_session_object_can_be_reused(mocker):
    fakes = [_FakeAsyncServer(), _FakeAsyncServer()]

    async def connect(uri, additional_headers):
        return fakes.pop(0)

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )
    session = aai.AsyncTTSSession()
    async with session:
        assert b"".join(await _drain(session.speak("One."))) == b"One."
    async with session:
        assert b"".join(await _drain(session.speak("Two."))) == b"Two."


async def test_a_cancelled_connect_leaves_nothing_open(mocker):
    slow = _FakeAsyncServer(begin=None)  # Begin never comes
    fakes = [slow, _FakeAsyncServer()]

    async def connect(uri, additional_headers):
        return fakes.pop(0)

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )
    session = aai.AsyncTTSSession()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(session.connect(), 0.3)

    assert slow.closed
    assert session._reader is not None and session._reader.done()
    begin = await asyncio.wait_for(session.connect(), 5)  # not "already connected"
    assert begin.id == "sess-1"
    await session.close()


async def test_a_cancelled_terminate_still_closes(server):
    server.answer_terminate = False
    session = aai.AsyncTTSSession(terminate_timeout=30)
    await session.connect()
    ending = asyncio.create_task(session.terminate())
    await asyncio.sleep(0.05)
    ending.cancel()
    await asyncio.wait({ending}, timeout=5)

    assert ending.cancelled()
    assert server.closed
    assert session._ws is None


async def test_a_cancelled_close_propagates_the_cancellation(server):
    in_handler = asyncio.Event()

    async def busy_handler(event):
        in_handler.set()
        await asyncio.sleep(30)

    session = aai.AsyncTTSSession(terminate_timeout=30)
    await session.connect()
    session.on(aai.TTSEvents.FlushDone, busy_handler)
    await session.flush()
    await asyncio.wait_for(in_handler.wait(), 5)

    closing = asyncio.create_task(session.close())
    await asyncio.sleep(0.05)  # close() now waits for the busy reader
    closing.cancel()
    await asyncio.wait({closing}, timeout=5)

    assert closing.cancelled()
    await asyncio.sleep(0)
    assert session._reader is not None and session._reader.done()


def _connect_to(mocker, *fakes: "_FakeAsyncServer") -> None:
    queued = list(fakes)

    async def connect(uri, additional_headers):
        return queued.pop(0)

    mocker.patch(
        "assemblyai.tts.v1.async_session.websocket_connect_async", side_effect=connect
    )


async def test_connect_after_the_api_ended_the_session_opens_a_new_one(mocker):
    first, second = _FakeAsyncServer(), _FakeAsyncServer()
    _connect_to(mocker, first, second)
    session = aai.AsyncTTSSession()
    await session.connect()
    first.push(_INACTIVITY_TIMEOUT)
    first.push(
        ConnectionClosedError(Close(3006, "See Error message for details"), None)
    )
    await asyncio.wait_for(session._stopped.wait(), 5)
    assert session.error is not None and session.error.retryable is True

    await session.connect()  # no close() needed first
    assert first.closed
    assert session.error is None
    assert b"".join(await _drain(session.speak("Hello again."))) == b"Hello again."
    await session.terminate()


async def test_connect_on_a_live_session_is_refused(server):
    async with aai.AsyncTTSSession() as session:
        with pytest.raises(RuntimeError, match="already connected"):
            await session.connect()


async def test_connect_right_after_close_gave_up_on_a_handler(mocker):
    first, second = _FakeAsyncServer(), _FakeAsyncServer()
    _connect_to(mocker, first, second)
    in_handler = asyncio.Event()

    async def busy_handler(event):
        in_handler.set()
        await asyncio.sleep(30)

    session = aai.AsyncTTSSession(terminate_timeout=0.1)
    await session.connect()
    session.on(aai.TTSEvents.Cancelled, busy_handler)
    await session.cancel()
    await asyncio.wait_for(in_handler.wait(), 5)
    await session.close()  # gives up on the busy reader and cancels it

    await asyncio.wait_for(session.connect(), 5)  # at once
    await asyncio.sleep(0.05)
    assert not session._stopped.is_set()
    assert b"".join(await _drain(session.speak("Hi."))) == b"Hi."
    await session.terminate()


async def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setattr(aai.settings, "api_key", None)
    with pytest.raises(ValueError, match="api_key= or token="):
        aai.AsyncTTSSession()
