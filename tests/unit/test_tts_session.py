"""Tests for the threaded Text-to-Speech WebSocket session.

`_FakeServer` plays the API: it answers `Generate`/`Flush` with the text's
bytes as one `Audio` frame and a `FlushDone`, `Cancel` with `Cancelled`, and
`Terminate` with `Termination` and a clean close.
"""

import base64
import json
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

import assemblyai as aai

aai.settings.api_key = "test"

_BEGIN = {
    "type": "Begin",
    "id": "sess-1",
    "expires_at": 1789556741,
    "configuration": {
        "voice_id": "aai-jane",
        "model": "assemblyai-tts-1",
        "sample_rate": 24000,
    },
}
_TERMINATION = {
    "type": "Termination",
    "session_duration_seconds": 3,
    "total_input_char_length": 20,
    "audio_duration_seconds": 1.5,
}


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
    pytest.param(ValueError("not a WebSocket URI"), None, False, id="bad-uri"),
]


def _closed_ok() -> ConnectionClosedOK:
    return ConnectionClosedOK(Close(1000, "Session ended"), None)


class _FakeServer:
    def __init__(self, begin: Optional[dict] = _BEGIN, answer_flush: bool = True):
        self.sent: List[dict] = []
        self.inbox: "queue.Queue[Any]" = queue.Queue()
        self.closed = False
        self.text = ""
        self.flush_id = 0
        self.answer_flush = answer_flush
        self.generate_delay = 0.0
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
        self.inbox.put(json.dumps(item) if isinstance(item, dict) else item)

    def send(self, raw: str) -> None:
        if self.closing is not None:
            closing, self.closing = self.closing, None
            for item in closing:
                self.push(item)
            raise ConnectionClosedError(closing[-1].rcvd, None)
        frame = json.loads(raw)
        self.sent.append(frame)
        kind = frame["type"]
        if kind == "Generate":
            time.sleep(self.generate_delay)
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
            self.push(_closed_ok())

    def recv(self, timeout: Optional[float] = None) -> str:
        try:
            item = self.inbox.get(timeout=0.02)
        except queue.Empty:
            raise TimeoutError()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def server(mocker):
    fake = _FakeServer()
    connect = mocker.patch(
        "assemblyai.tts.v1.session.websocket_connect", return_value=fake
    )
    fake.connect = connect
    return fake


def _drain(items: Iterable[Any], timeout: float = 5.0) -> List[Any]:
    """Collects `items` on a thread, failing the test if it does not end."""

    got: List[Any] = []
    raised: List[BaseException] = []

    def collect() -> None:
        try:
            got.extend(items)
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"the iteration did not end within {timeout} s")
    if raised:
        raise raised[0]
    return got


def _frames(server: _FakeServer, kind: str) -> List[dict]:
    return [f for f in server.sent if f["type"] == kind]


def test_with_connects_waits_for_begin_and_terminates_on_exit(server):
    with aai.TTSSession(
        config=aai.TTSConfig(voice_id="aai-alba", sample_rate=8000)
    ) as session:
        assert session.begin is not None
        assert session.begin.id == "sess-1"
        assert session.begin.configuration["model"] == "assemblyai-tts-1"

    uri = server.connect.call_args.args[0]
    headers = server.connect.call_args.kwargs["additional_headers"]
    parts = urlsplit(uri)
    assert parts.scheme == "wss" and parts.netloc == "tts.us.assemblyai.com"
    assert parts.path == "/v1/ws"
    assert parse_qs(parts.query) == {
        "voice_id": ["aai-alba"],
        "sample_rate": ["8000"],
        "word_boundaries": ["false"],
    }
    assert headers["Authorization"] == "test"
    assert "AssemblyAI" in headers["User-Agent"]
    assert [f["type"] for f in server.sent] == ["Terminate"]
    assert session.termination is not None
    assert session.termination.total_input_char_length == 20
    assert server.closed


def test_speak_sends_generate_and_flush_and_yields_the_audio(server):
    with aai.TTSSession() as session:
        chunks = _drain(session.speak("Hello there."))

    assert b"".join(chunks) == b"Hello there."
    assert server.sent[:2] == [
        {"type": "Generate", "text": "Hello there."},
        {"type": "Flush"},
    ]


def test_requests_are_pipelined_and_read_in_flush_id_order(server):
    with aai.TTSSession() as session:
        session.generate("One. ")
        first = session.flush()
        session.generate("Two.")
        second = session.flush()
        assert (first, second) == (0, 1)
        assert b"".join(_drain(session.audio(first))) == b"One. "
        assert b"".join(_drain(session.audio(second))) == b"Two."


def test_cancel_ends_the_current_request_but_not_a_later_one(server):
    server.answer_flush = False  # the request stays in flight
    with aai.TTSSession() as session:
        session.generate("A long reply that gets interrupted.")
        flush_id = session.flush()
        stream = session.audio(flush_id)
        session.cancel()  # barge-in
        assert _drain(stream) == []
        assert session._next_flush_id == 2

        server.answer_flush = True
        assert b"".join(_drain(session.speak("Next."))) == b"Next."
    assert [f["type"] for f in server.sent] == [
        "Generate",
        "Flush",
        "Cancel",
        "Generate",
        "Flush",
        "Terminate",
    ]


def test_a_cancel_from_another_thread_keeps_flush_ids_in_wire_order(server):
    server.generate_delay = 0.02  # a slow write the cancel lands in
    with aai.TTSSession() as session:
        for _ in range(5):
            flush_ids: List[int] = []

            def write_reply() -> None:
                session.generate("Hello there. ")
                flush_ids.append(session.flush())

            writer = threading.Thread(target=write_reply)
            writer.start()
            time.sleep(0.01)
            session.cancel()  # barge-in from a VAD callback, say
            writer.join(5)
            _drain(session.audio(flush_ids[0]))  # ends, cancelled or spoken

        # every Flush and Cancel took the id the API gives it
        assert session._next_flush_id == server.flush_id == 10


def test_cancel_ends_the_iteration_before_the_server_answers(server):
    server.answer_flush = False
    with aai.TTSSession() as session:
        flush_id = session.flush()
        server.push({"type": "Audio", "audio": "AA==", "flush_id": 0})
        stream = session.audio(flush_id)
        assert next(stream) == b"\x00"
        session.cancel()
        # in-transit audio of the cancelled request is discarded, not yielded
        server.push({"type": "Audio", "audio": "AA==", "flush_id": 0})
        assert _drain(stream) == []


def test_audio_for_an_unflushed_id_raises(server):
    with aai.TTSSession() as session:
        with pytest.raises(ValueError, match="not been flushed"):
            next(session.audio(5))


def test_audio_without_an_id_follows_the_request_being_written(server):
    with aai.TTSSession() as session:
        got = []
        reader = threading.Thread(target=lambda: got.extend(session.audio()))
        reader.start()  # reading before the flush, like a player thread
        session.generate("Early ")
        session.generate("audio.")
        session.flush()
        reader.join(timeout=5)
    assert b"".join(got) == b"Early audio."


def test_a_misspelled_event_name_is_refused(server):
    session = aai.TTSSession()
    with pytest.raises(ValueError, match="FlushDone"):
        session.on("FlushDon", print)
    session.on("FlushDone", print)  # a string spelling of a member is fine


def test_sending_before_connect_says_how_to_open(server):
    with pytest.raises(RuntimeError, match="with"):
        aai.TTSSession().generate("Hi.")


def test_audio_before_connect_says_the_session_is_not_open(server):
    with pytest.raises(RuntimeError, match="not open"):
        _drain(aai.TTSSession().audio())


def test_http_only_options_are_refused_on_a_session(server):
    with pytest.raises(ValueError, match="container and bit_rate"):
        with aai.TTSSession(config=aai.TTSConfig(container="wav")):
            pass


def test_a_cancel_before_a_request_does_not_end_it(server):
    with aai.TTSSession() as session:
        session.cancel()  # nothing in flight; Cancelled arrives later
        assert b"".join(_drain(session.speak("Hi."))) == b"Hi."


def test_an_error_before_begin_raises_from_with(mocker):
    fake = _FakeServer(begin=None)
    fake.push(
        {
            "type": "Error",
            "error_code": 3012,
            "error": "The requested voice was not found.",
            "error_type": "voice_not_found",
            "retryable": False,
            "param": "voice_id",
            "retry_after_seconds": None,
        }
    )
    fake.push(ConnectionClosedError(Close(3012, "See Error message for details"), None))
    mocker.patch("assemblyai.tts.v1.session.websocket_connect", return_value=fake)

    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession(config=aai.TTSConfig(voice_id="aai-nobody")):
            pass

    err = exc_info.value
    assert err.error_code == "voice_not_found"
    assert err.param == "voice_id"
    assert err.retryable is False
    assert err.body["error_code"] == 3012
    assert err.body["close_code"] == 3012  # the key a bare close uses too
    assert err.request_id is None  # no Begin, so no session id yet
    assert fake.closed


@pytest.mark.parametrize("failure, status, retryable", _HANDSHAKE_FAILURES)
def test_a_failed_handshake_keeps_its_status_and_says_if_retryable(
    mocker, failure, status, retryable
):
    mocker.patch("assemblyai.tts.v1.session.websocket_connect", side_effect=failure)
    with pytest.raises(aai.TTSError, match="handshake failed") as exc_info:
        with aai.TTSSession():
            pass
    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable
    assert exc_info.value.__cause__ is failure


def test_an_error_mid_session_raises_from_the_audio_and_reaches_the_handler(server):
    errors = []
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            session.on(aai.TTSEvents.Error, errors.append)
            server.answer_flush = False
            flush_id = session.flush()
            server.push(
                {
                    "type": "Error",
                    "error_code": 3005,
                    "error": "No room",
                    "error_type": "at_capacity",
                    "retryable": True,
                    "param": None,
                    "retry_after_seconds": 2,
                }
            )
            _drain(session.audio(flush_id))

    assert exc_info.value.error_code == "at_capacity"
    assert exc_info.value.retry_after == 2
    assert exc_info.value.request_id == "sess-1"  # the session id
    assert errors == [exc_info.value]
    assert session.error is exc_info.value


def test_a_lost_connection_is_a_retryable_error(server):
    errors = []
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            session.on(aai.TTSEvents.Error, errors.append)
            server.answer_flush = False
            flush_id = session.flush()
            server.push(ConnectionClosedError(None, None))  # abnormal: no close frame
            _drain(session.audio(flush_id))

    assert exc_info.value.retryable is True
    assert exc_info.value.body == {"close_code": 1006, "reason": ""}
    assert len(errors) == 1


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
def test_a_close_without_an_error_frame_is_retryable_by_its_code(
    server, code, retryable
):
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            server.answer_flush = False
            flush_id = session.flush()
            server.push(ConnectionClosedError(Close(code, "Upstream failed"), None))
            _drain(session.audio(flush_id))

    assert exc_info.value.retryable is retryable
    assert exc_info.value.body == {"close_code": code, "reason": "Upstream failed"}


def test_a_socket_error_is_a_retryable_lost_connection(server):
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            server.answer_flush = False
            flush_id = session.flush()
            server.push(OSError("Connection reset by peer"))
            _drain(session.audio(flush_id))

    assert exc_info.value.retryable is True
    assert exc_info.value.body == {"close_code": 1006, "reason": ""}


def test_a_failure_in_the_reader_still_ends_the_iteration(server, mocker):
    mocker.patch(
        "assemblyai.tts.v1.session.error_from_close", side_effect=RuntimeError("bug")
    )
    with pytest.raises(aai.TTSError, match="reader failed"):
        with aai.TTSSession() as session:
            server.answer_flush = False
            flush_id = session.flush()
            server.push(ConnectionClosedError(None, None))
            _drain(session.audio(flush_id))


def test_a_send_after_a_failure_raises_the_stored_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            server.push(ConnectionClosedError(None, None))
            session._stopped.wait(2)
            session.generate("too late")
    assert exc_info.value.body["close_code"] == 1006


def test_a_send_the_api_closed_raises_the_sessions_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            server.close_on_next_send(3010, _INPUT_RATE_EXCEEDED)
            session.generate("Too much, too fast.")

    assert exc_info.value.error_code == "input_rate_exceeded"
    assert exc_info.value.retryable is False
    assert exc_info.value is session.error


def test_a_send_on_a_bare_close_raises_a_tts_error(server):
    with pytest.raises(aai.TTSError) as exc_info:
        with aai.TTSSession() as session:
            server.close_on_next_send(1012)
            session.flush()

    assert exc_info.value.retryable is True
    assert exc_info.value.body["close_code"] == 1012


def test_a_cancel_drops_every_pipelined_request(server):
    server.answer_flush = False
    with aai.TTSSession() as session:
        first = session.flush()
        second = session.flush()
        session.cancel()
        assert _drain(session.audio(first)) == []
        assert _drain(session.audio(second)) == []  # also dropped, must not hang


def test_an_unreadable_frame_ends_the_session_with_an_error(server):
    with pytest.raises(aai.TTSError, match="Unreadable frame"):
        with aai.TTSSession() as session:
            server.answer_flush = False
            flush_id = session.flush()
            server.push({"type": "Audio", "flush_id": "zero"})  # no audio, wrong type
            _drain(session.audio(flush_id))


def test_a_session_object_can_be_reused(mocker):
    fakes = [_FakeServer(), _FakeServer()]
    mocker.patch("assemblyai.tts.v1.session.websocket_connect", side_effect=fakes)
    session = aai.TTSSession()
    with session:
        assert b"".join(_drain(session.speak("One."))) == b"One."
    with session:
        assert b"".join(_drain(session.speak("Two."))) == b"Two."
    assert [f["type"] for f in fakes[1].sent] == ["Generate", "Flush", "Terminate"]


def test_connect_after_the_api_ended_the_session_opens_a_new_one(mocker):
    first, second = _FakeServer(), _FakeServer()
    mocker.patch(
        "assemblyai.tts.v1.session.websocket_connect", side_effect=[first, second]
    )
    session = aai.TTSSession()
    session.connect()
    first.push(_INACTIVITY_TIMEOUT)
    first.push(
        ConnectionClosedError(Close(3006, "See Error message for details"), None)
    )
    assert session._stopped.wait(5)
    assert session.error is not None and session.error.retryable is True

    session.connect()  # no close() needed first
    assert first.closed
    assert session.error is None
    assert b"".join(_drain(session.speak("Hello again."))) == b"Hello again."
    session.terminate()


def test_connect_on_a_live_session_is_refused(server):
    with aai.TTSSession() as session:
        with pytest.raises(RuntimeError, match="already connected"):
            session.connect()


def test_a_new_connection_waits_for_the_previous_connections_handler(mocker):
    first, second = _FakeServer(), _FakeServer()
    mocker.patch(
        "assemblyai.tts.v1.session.websocket_connect", side_effect=[first, second]
    )
    in_handler = threading.Event()

    def blocking_player(event: aai.TTSAudioEvent) -> None:
        in_handler.set()
        time.sleep(1)  # a playback write that blocks

    session = aai.TTSSession(terminate_timeout=0.2)
    session.on(aai.TTSEvents.Audio, blocking_player)
    with session:
        session.generate("Hi.")
        session.flush()
        assert in_handler.wait(5)
    # the previous reader is still in the handler here
    with session:
        time.sleep(1.5)  # the previous handler has returned by now
        session.generate("Still open.")
        assert not session._stopped.is_set()
        assert session.termination is None


def test_unread_audio_is_bounded_and_dropped_oldest_first(server, caplog):
    from assemblyai.tts._base_ws import PULL_QUEUE_FRAMES

    with aai.TTSSession() as session:
        # audio of a request nobody reads floods the pull queue
        for _ in range(PULL_QUEUE_FRAMES + 5):
            server.push({"type": "Audio", "audio": "AA==", "flush_id": 7})
        deadline = time.monotonic() + 5
        while not server.inbox.empty() and time.monotonic() < deadline:
            time.sleep(0.01)  # let the reader queue the flood before anyone reads
        assert len(session._queue) == PULL_QUEUE_FRAMES
        session.generate("y")
        flush_id = session.flush()
        # still served after the flood
        assert b"".join(_drain(session.audio(flush_id))) == b"y"
    assert session._queue.max_audio == PULL_QUEUE_FRAMES
    assert "oldest are being dropped" in caplog.text


def test_an_overflow_drops_audio_but_never_the_end_of_a_request(server, mocker, caplog):
    mocker.patch("assemblyai.tts.v1.session.PULL_QUEUE_FRAMES", 6)
    server.answer_flush = False
    done: List[int] = []
    with aai.TTSSession() as session:
        session.on(aai.TTSEvents.FlushDone, lambda event: done.append(event.flush_id))
        flush_ids = [session.flush() for _ in range(3)]
        for flush_id in flush_ids:  # 4 frames each, all before anything is read
            for _ in range(4):
                server.push({"type": "Audio", "audio": "AA==", "flush_id": flush_id})
            server.push(
                {"type": "FlushDone", "flush_id": flush_id, "audio_duration_ms": 320}
            )
        deadline = time.monotonic() + 5
        while len(done) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)

        got = [_drain(session.audio(flush_id)) for flush_id in flush_ids]

    # 12 frames for room for 6: the oldest 6 went, every request still ended
    assert [len(chunks) for chunks in got] == [0, 2, 4]
    assert "oldest are being dropped" in caplog.text


def test_an_audio_handler_receives_the_frames_instead_of_the_queue(server):
    received = []
    done = []
    with aai.TTSSession() as session:
        session.on(aai.TTSEvents.Audio, received.append)
        session.on(aai.TTSEvents.FlushDone, done.append)
        session.generate("Pushed.")
        flush_id = session.flush()
        with pytest.raises(RuntimeError, match="Audio handler"):
            _drain(session.audio(flush_id))
        session._stopped.wait(0.2)  # let the frames land

    assert [e.audio for e in received] == [b"Pushed."]
    assert received[0].flush_id == 0
    assert done[0].audio_duration_ms == 80


def test_the_error_reaches_waiters_before_the_error_handler(server):
    app_saw_error = threading.Event()
    handler_waited: List[bool] = []

    def on_error(error: aai.TTSError) -> None:
        handler_waited.append(app_saw_error.wait(5))  # waits on the app

    with pytest.raises(aai.TTSError):
        with aai.TTSSession() as session:
            session.on(aai.TTSEvents.Error, on_error)
            server.answer_flush = False
            flush_id = session.flush()
            server.push(_INPUT_RATE_EXCEEDED)
            try:
                _drain(session.audio(flush_id), timeout=3)
            finally:
                app_saw_error.set()

    assert handler_waited == [True]


def test_waiting_from_a_handler_raises_instead_of_deadlocking(server):
    refused: List[str] = []

    def speak_next(event: aai.TTSFlushDoneEvent) -> None:
        for wait in (lambda: list(session.speak("Next.")), session.terminate):
            try:
                wait()
            except RuntimeError as exc:
                refused.append(str(exc))

    with aai.TTSSession(terminate_timeout=3) as session:
        session.on(aai.TTSEvents.FlushDone, speak_next)
        assert b"".join(_drain(session.speak("Hi."))) == b"Hi."

    assert len(refused) == 2
    assert all("from a session handler" in message for message in refused)
    assert _frames(server, "Flush") == [{"type": "Flush"}]  # nothing more was sent


def test_word_boundaries_and_cancelled_reach_their_handlers(server):
    words = []
    cancelled = []
    with aai.TTSSession(word_boundaries=True) as session:
        session.on(aai.TTSEvents.WordBoundaries, words.append)
        session.on(aai.TTSEvents.Cancelled, cancelled.append)
        _drain(session.speak("Hello there."))
        server.push(
            {
                "type": "WordBoundaries",
                "flush_id": 0,
                "audio_start_ms": 0,
                "words": [
                    {"word": "Hello", "start": 265, "end": 530, "confidence": 0.9}
                ],
            }
        )
        session.cancel()
        session.keep_alive()
        _drain(session.speak("Again."))

    assert parse_qs(urlsplit(server.connect.call_args.args[0]).query)[
        "word_boundaries"
    ] == ["true"]
    assert words[0].words[0].word == "Hello"
    assert cancelled[0].audio_duration_seconds == 0.1
    assert [f["type"] for f in server.sent].count("KeepAlive") == 1


def test_a_temporary_token_goes_in_the_query_and_not_the_header(server, monkeypatch):
    monkeypatch.setattr(aai.settings, "api_key", None)  # a token needs no key
    with aai.TTSSession(token="tmp-token", inactivity_timeout=30):
        pass

    uri = server.connect.call_args.args[0]
    headers = server.connect.call_args.kwargs["additional_headers"]
    query = parse_qs(urlsplit(uri).query)
    assert query["token"] == ["tmp-token"]
    assert query["inactivity_timeout"] == ["30"]
    assert "Authorization" not in headers


def test_generate_refuses_text_over_one_frame(server):
    with aai.TTSSession() as session:
        with pytest.raises(ValueError, match="2000"):
            session.generate("x" * 2001)
        with pytest.raises(TypeError):
            session.generate(b"bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "limit, allowed, refused", [(4000, 4000, 4001), (500, 500, 501)]
)
def test_generate_follows_the_limit_begin_reports(mocker, limit, allowed, refused):
    begin = dict(_BEGIN, configuration={"limits": {"max_generate_text_length": limit}})
    mocker.patch(
        "assemblyai.tts.v1.session.websocket_connect",
        return_value=_FakeServer(begin=begin),
    )
    with aai.TTSSession() as session:
        session.generate("x" * allowed)
        with pytest.raises(ValueError, match=f"{limit}-character"):
            session.generate("x" * refused)


def test_the_eu_host_comes_from_the_settings(server):
    client = aai.Client(api_key="test")
    client.settings.tts_base_url = "https://tts.eu.assemblyai.com"
    with aai.TTSSession(client=client):
        pass
    assert urlsplit(server.connect.call_args.args[0]).netloc == "tts.eu.assemblyai.com"


def test_a_plain_http_host_opens_a_plain_websocket(server):
    client = aai.Client(api_key="test")
    client.settings.tts_base_url = "http://localhost:8080"
    with aai.TTSSession(client=client):
        pass
    assert server.connect.call_args.args[0].startswith("ws://localhost:8080/v1/ws?")


def test_api_key_argument_is_used_without_touching_the_shared_settings(server):
    with aai.TTSSession(
        api_key="other", config=aai.TTSConfig(extend_biometric_consent=True)
    ):
        pass
    headers = server.connect.call_args.kwargs["additional_headers"]
    query = parse_qs(urlsplit(server.connect.call_args.args[0]).query)
    assert headers["Authorization"] == "other"
    assert query["extend_biometric_consent"] == ["true"]
    assert aai.settings.api_key == "test"


@pytest.mark.parametrize("configuration", [None, "missing"])
def test_a_begin_without_configuration_opens_the_session(mocker, configuration):
    begin = {k: v for k, v in _BEGIN.items() if k != "configuration"}
    if configuration != "missing":
        begin["configuration"] = configuration
    mocker.patch(
        "assemblyai.tts.v1.session.websocket_connect",
        return_value=_FakeServer(begin=begin),
    )
    with aai.TTSSession() as session:
        assert session.begin is not None and session.begin.configuration == {}
        assert b"".join(_drain(session.speak("Hi."))) == b"Hi."


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setattr(aai.settings, "api_key", None)
    with pytest.raises(ValueError, match="api_key= or token="):
        aai.TTSSession()


def test_an_exception_inside_with_closes_without_terminate(server):
    with pytest.raises(RuntimeError):
        with aai.TTSSession():
            raise RuntimeError("boom")
    assert server.sent == []
    assert server.closed


def test_wrong_config_type_is_a_type_error():
    with pytest.raises(TypeError):
        aai.TTSSession(config=aai.DictationConfig())  # type: ignore[arg-type]
