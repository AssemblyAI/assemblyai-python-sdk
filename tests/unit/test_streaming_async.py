import asyncio
import json
import logging
from urllib.parse import urlencode

import pytest
from pytest_mock import MockFixture
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.frames import Close

from assemblyai.streaming.v3 import (
    AsyncStreamingClient,
    SpeechModel,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)
from assemblyai.streaming.v3.models import TerminateSession

pytestmark = pytest.mark.asyncio


def _default_params() -> StreamingParameters:
    return StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )


class _FakeAsyncWebSocket:
    """Programmable async websocket stand-in for driving AsyncStreamingClient
    in tests. Inbound messages are queued via ``push_message`` /
    ``push_close``; outbound sends accumulate in ``sent``.
    """

    def __init__(self, send_raises=None):
        self._inbound: "asyncio.Queue[object]" = asyncio.Queue()
        self._send_raises = send_raises
        self.sent: list = []
        self.send_call_count = 0
        self.close_call_count = 0
        self._closed = False

    def push_message(self, data) -> None:
        self._inbound.put_nowait(data)

    def push_close(self, exc: BaseException) -> None:
        self._inbound.put_nowait(exc)

    async def recv(self):
        item = await self._inbound.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, data) -> None:
        self.send_call_count += 1
        if self._send_raises is not None:
            raise self._send_raises
        self.sent.append(data)

    async def close(self) -> None:
        self.close_call_count += 1
        self._closed = True


def _patch_connect(mocker: MockFixture, fake_ws):
    """Patch ``websocket_connect_async`` to return the given fake websocket."""

    async def fake_connect(uri, additional_headers=None, **_kwargs):
        fake_connect.uri = uri
        fake_connect.additional_headers = additional_headers
        return fake_ws

    fake_connect.uri = None
    fake_connect.additional_headers = None
    mocker.patch(
        "assemblyai.streaming.v3.async_client.websocket_connect_async",
        new=fake_connect,
    )
    return fake_connect


async def _wait_for_tasks(client: AsyncStreamingClient, timeout: float = 2.0) -> None:
    """Wait until both read/write tasks have exited and stop is set. Raises
    ``AssertionError`` on timeout so stalls fail tests deterministically
    instead of silently passing."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        read_done = client._read_task is None or client._read_task.done()
        write_done = client._write_task is None or client._write_task.done()
        if read_done and write_done and client._stop_event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"AsyncStreamingClient read/write tasks did not finish within {timeout}s"
    )


async def test_client_connect_builds_uri_and_headers(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    fake_connect = _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    params = _default_params()
    await client.connect(params)

    expected_qs = urlencode(
        {
            "sample_rate": params.sample_rate,
            "speech_model": str(params.speech_model),
        }
    )
    assert fake_connect.uri == f"wss://api.example.com/v3/ws?{expected_qs}"
    assert fake_connect.additional_headers["Authorization"] == "test"
    assert fake_connect.additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in fake_connect.additional_headers["User-Agent"]

    await client.disconnect()


async def test_client_connect_with_token(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    fake_connect = _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(token="tok-value", api_host="api.example.com")
    )
    await client.connect(_default_params())

    assert fake_connect.additional_headers["Authorization"] == "tok-value"

    await client.disconnect()


async def test_stream_bytes_writes_to_socket(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    await client.stream(b"\x00" * 320)

    # Give the write task a moment to drain the queue.
    for _ in range(50):
        if fake_ws.sent:
            break
        await asyncio.sleep(0.01)

    assert fake_ws.sent == [b"\x00" * 320]

    await client.disconnect()


async def test_stream_sync_iterable(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    chunks = [b"a", b"bb", b"ccc"]
    await client.stream(iter(chunks))

    for _ in range(50):
        if len(fake_ws.sent) == 3:
            break
        await asyncio.sleep(0.01)

    assert fake_ws.sent == chunks

    await client.disconnect()


async def test_stream_async_iterable(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    async def gen():
        for chunk in (b"x", b"yy", b"zzz"):
            yield chunk

    await client.stream(gen())

    for _ in range(50):
        if len(fake_ws.sent) == 3:
            break
        await asyncio.sleep(0.01)

    assert fake_ws.sent == [b"x", b"yy", b"zzz"]

    await client.disconnect()


async def test_disconnect_terminate_sends_terminate_then_closes(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    await client.disconnect(terminate=True)

    sent_terminate = [
        s for s in fake_ws.sent if isinstance(s, str) and "Terminate" in s
    ]
    assert len(sent_terminate) == 1
    assert fake_ws.close_call_count >= 1


async def test_begin_event_dispatched_to_handler(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_begin(_client, event):
        received.append(event)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps(
            {
                "type": "Begin",
                "id": "abc",
                "expires_at": "2030-01-01T00:00:00",
            }
        )
    )

    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].id == "abc"

    await client.disconnect()


async def test_async_handler_is_awaited(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    seen = []

    async def on_begin(_client, event):
        await asyncio.sleep(0)
        seen.append(event.id)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps(
            {"type": "Begin", "id": "async-id", "expires_at": "2030-01-01T00:00:00"}
        )
    )

    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.01)

    assert seen == ["async-id"]

    await client.disconnect()


async def test_sync_and_async_handlers_can_mix(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    sync_seen = []
    async_seen = []

    def sync_handler(_client, event):
        sync_seen.append(event.id)

    async def async_handler(_client, event):
        async_seen.append(event.id)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Begin, sync_handler)
    client.on(StreamingEvents.Begin, async_handler)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps({"type": "Begin", "id": "mix", "expires_at": "2030-01-01T00:00:00"})
    )

    for _ in range(50):
        if sync_seen and async_seen:
            break
        await asyncio.sleep(0.01)

    assert sync_seen == ["mix"]
    assert async_seen == ["mix"]

    await client.disconnect()


async def test_error_event_then_close_fires_only_once(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_error(_client, err):
        received.append(err)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps({"type": "Error", "error": "Invalid API key", "error_code": 4001})
    )
    fake_ws.push_close(ConnectionClosed(rcvd=Close(4001, "Not Authorized"), sent=None))

    await _wait_for_tasks(client)

    assert len(received) == 1
    assert str(received[0]) == "Invalid API key"
    assert received[0].code == 4001

    error_logs = [
        rec
        for rec in caplog.records
        if "Streaming error" in rec.message and "4001" in rec.message
    ]
    close_logs = [
        rec
        for rec in caplog.records
        if "Connection closed" in rec.message and "4001" in rec.message
    ]
    assert len(error_logs) == 1
    # ``_report_server_error`` closes the websocket locally and sets stop, so
    # the read loop exits before the pushed trailing close is recv'd. No close
    # log is emitted in this path — the Error event already captured the cause.
    assert close_logs == []

    await client.disconnect()


async def test_server_error_without_trailing_close_tears_down(mocker: MockFixture):
    """Regression: a server ``Error`` frame with no trailing close must still
    drive the read loop to exit. Without local teardown in
    ``_report_server_error``, ``await ws.recv()`` would block indefinitely."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_error(_client, err):
        received.append(err)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)
    await client.connect(_default_params())

    # Push an Error frame and nothing else — no trailing close.
    fake_ws.push_message(
        json.dumps({"type": "Error", "error": "boom", "error_code": 4002})
    )

    # If teardown is missing this raises AssertionError after timeout.
    await _wait_for_tasks(client)

    assert len(received) == 1
    assert received[0].code == 4002
    assert fake_ws.close_call_count >= 1

    await client.disconnect()


async def test_clean_close_emits_no_error_or_log(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_error(_client, err):
        received.append(err)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)
    await client.connect(_default_params())

    fake_ws.push_close(ConnectionClosed(rcvd=Close(1000, "session ended"), sent=None))

    await _wait_for_tasks(client)

    assert received == []
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert error_logs == []

    await client.disconnect()


async def test_turn_handler_exception_does_not_kill_read_task(mocker: MockFixture):
    """A raising Turn handler must not propagate out of the read task; the
    next inbound message should still be delivered."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    seen = []

    def bad_handler(_client, _turn):
        raise RuntimeError("boom")

    def good_handler(_client, turn):
        seen.append(turn.end_of_turn)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Turn, bad_handler)
    client.on(StreamingEvents.Turn, good_handler)
    await client.connect(_default_params())

    turn_payload = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": False,
        "end_of_turn": False,
        "transcript": "hello",
        "end_of_turn_confidence": 0.5,
        "words": [],
    }
    fake_ws.push_message(json.dumps(turn_payload))
    fake_ws.push_message(json.dumps({**turn_payload, "turn_order": 2}))

    for _ in range(100):
        if len(seen) == 2:
            break
        await asyncio.sleep(0.01)

    assert seen == [False, False]

    await client.disconnect()


async def test_warning_handler_exception_does_not_kill_read_task(mocker: MockFixture):
    """A raising Warning handler must not propagate out of the read task."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def bad_handler(_client, _warning):
        raise RuntimeError("boom")

    def good_handler(_client, warning):
        received.append(warning.warning_code)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Warning, bad_handler)
    client.on(StreamingEvents.Warning, good_handler)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps({"type": "Warning", "warning": "first", "warning_code": 1})
    )
    fake_ws.push_message(
        json.dumps({"type": "Warning", "warning": "second", "warning_code": 2})
    )

    for _ in range(100):
        if len(received) == 2:
            break
        await asyncio.sleep(0.01)

    assert received == [1, 2]

    await client.disconnect()


async def test_stream_before_connect_raises_runtime_error():
    """``stream()`` called before ``connect()`` must raise RuntimeError rather
    than silently dropping data. Silent drop would diverge from the sync client
    (which buffers pre-connect data) in a way that's easy to miss — explicit
    failure surfaces the misuse."""
    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    async def gen():
        yield b"x"

    for data in (b"\x00" * 10, iter([b"a", b"b"]), gen()):
        with pytest.raises(RuntimeError, match="not connected"):
            await client.stream(data)


async def test_set_params_before_connect_raises_runtime_error():
    from assemblyai.streaming.v3 import (
        StreamingSessionParameters,
    )

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    with pytest.raises(RuntimeError, match="not connected"):
        await client.set_params(StreamingSessionParameters(min_turn_silence=200))


async def test_force_endpoint_before_connect_raises_runtime_error():
    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    with pytest.raises(RuntimeError, match="not connected"):
        await client.force_endpoint()


async def test_stream_after_close_is_noop(mocker: MockFixture):
    """Post-close ``stream()`` must stay a silent no-op so user cleanup paths
    (e.g. a finally block draining a queue) don't have to wrap each call in
    try/except. Pre-connect raise + post-close no-op gives both: misuse is
    loud, cleanup is quiet."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    # Simulate a clean close — read task exits, _stop_event is set.
    fake_ws.push_close(ConnectionClosed(rcvd=Close(1000, "bye"), sent=None))
    await _wait_for_tasks(client)

    # No raise: post-close stream is safe for cleanup.
    await client.stream(b"\x00" * 10)
    await client.disconnect()


async def test_handler_exception_does_not_block_shutdown(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    def bad_handler(_client, _err):
        raise RuntimeError("boom")

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, bad_handler)
    await client.connect(_default_params())

    fake_ws.push_close(ConnectionClosed(rcvd=Close(1011, "server error"), sent=None))

    await _wait_for_tasks(client)
    # If the handler exception had escaped, _wait_for_tasks would time out.
    assert client._read_task.done()

    await client.disconnect()


async def test_invalid_status_during_connect_dispatches_error(mocker: MockFixture):
    received = []

    def on_error(_client, err):
        received.append(err)

    response = type("R", (), {"status_code": 401})()
    err = InvalidStatus(response=response)

    async def failing_connect(*_args, **_kwargs):
        raise err

    mocker.patch(
        "assemblyai.streaming.v3.async_client.websocket_connect_async",
        new=failing_connect,
    )

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)

    await client.connect(_default_params())

    assert len(received) == 1
    assert received[0].code == 401
    assert "HTTP 401" in str(received[0])


async def test_terminate_session_bypasses_stop_gate(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    # Pre-set stop, then queue a TerminateSession directly. The write loop must
    # still send it before exiting.
    client._stop_event.set()
    await client._write_queue.put(TerminateSession())

    for _ in range(100):
        if fake_ws.send_call_count >= 1:
            break
        await asyncio.sleep(0.01)

    assert fake_ws.send_call_count >= 1
    assert any(isinstance(s, str) and "Terminate" in s for s in fake_ws.sent)

    await client.disconnect()


async def test_create_temporary_token(mocker: MockFixture):
    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    captured = {}

    async def fake_get(self, url, params=None):
        captured["url"] = url
        captured["params"] = params

        class R:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"token": "tmp-tok"}

        return R()

    mocker.patch("httpx.AsyncClient.get", new=fake_get)

    token = await client.create_temporary_token(
        expires_in_seconds=60, max_session_duration_seconds=600
    )
    assert token == "tmp-tok"
    assert captured["url"] == "/v3/token"
    assert captured["params"] == {
        "expires_in_seconds": 60,
        "max_session_duration_seconds": 600,
    }

    await client._client.aclose()


async def test_create_temporary_token_forwards_zero_expires(mocker: MockFixture):
    """Regression: ``expires_in_seconds=0`` must reach the server (so it can
    reject it with a clear error) rather than being silently dropped by a
    falsy check."""
    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    captured = {}

    async def fake_get(self, url, params=None):
        captured["params"] = params

        class R:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"token": "tmp-tok"}

        return R()

    mocker.patch("httpx.AsyncClient.get", new=fake_get)

    await client.create_temporary_token(expires_in_seconds=0)

    assert captured["params"] == {"expires_in_seconds": 0}

    await client._client.aclose()


async def test_connect_twice_raises(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    with pytest.raises(RuntimeError, match="already been connected"):
        await client.connect(_default_params())

    await client.disconnect()


async def test_connect_after_handshake_failure_raises(mocker: MockFixture):
    """Regression: a failed connect leaves ``_connection_closed_reported`` set
    and ``_stop_event`` set. A second ``connect()`` attempt on the same client
    must surface a clear error, not silently produce a dead read/write loop."""
    response = type("R", (), {"status_code": 401})()
    err = InvalidStatus(response=response)

    async def failing_connect(*_args, **_kwargs):
        raise err

    mocker.patch(
        "assemblyai.streaming.v3.async_client.websocket_connect_async",
        new=failing_connect,
    )

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    await client.connect(_default_params())

    with pytest.raises(RuntimeError, match="already been connected"):
        await client.connect(_default_params())


async def test_set_params_enqueues_update_configuration(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    from assemblyai.streaming.v3.models import (
        StreamingSessionParameters,
    )

    await client.set_params(
        StreamingSessionParameters(end_of_turn_confidence_threshold=0.5)
    )

    for _ in range(100):
        update_frames = [
            s for s in fake_ws.sent if isinstance(s, str) and "UpdateConfiguration" in s
        ]
        if update_frames:
            break
        await asyncio.sleep(0.01)

    update_frames = [
        s for s in fake_ws.sent if isinstance(s, str) and "UpdateConfiguration" in s
    ]
    assert len(update_frames) == 1
    payload = json.loads(update_frames[0])
    assert payload["type"] == "UpdateConfiguration"
    assert payload["end_of_turn_confidence_threshold"] == 0.5

    await client.disconnect()


async def test_force_endpoint_enqueues_force_endpoint_frame(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    await client.force_endpoint()

    for _ in range(100):
        force_frames = [
            s for s in fake_ws.sent if isinstance(s, str) and "ForceEndpoint" in s
        ]
        if force_frames:
            break
        await asyncio.sleep(0.01)

    force_frames = [
        s for s in fake_ws.sent if isinstance(s, str) and "ForceEndpoint" in s
    ]
    assert len(force_frames) == 1
    payload = json.loads(force_frames[0])
    assert payload["type"] == "ForceEndpoint"

    await client.disconnect()


async def test_warning_event_dispatched_to_handler(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_warning(_client, event):
        received.append(event)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Warning, on_warning)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps({"type": "Warning", "warning": "slow audio", "warning_code": 1234})
    )

    for _ in range(100):
        if received:
            break
        await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].warning == "slow audio"
    assert received[0].warning_code == 1234

    await client.disconnect()


async def test_termination_event_sets_stop_and_dispatches(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    received = []

    def on_termination(_client, event):
        received.append(event)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Termination, on_termination)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps(
            {
                "type": "Termination",
                "audio_duration_seconds": 12,
                "session_duration_seconds": 15,
            }
        )
    )

    # Termination sets stop_event but doesn't close the socket; wait for the
    # handler to fire and stop_event to flip.
    for _ in range(100):
        if received and client._stop_event is not None and client._stop_event.is_set():
            break
        await asyncio.sleep(0.01)

    assert len(received) == 1
    assert client._stop_event is not None
    assert client._stop_event.is_set()

    await client.disconnect()


async def test_disconnect_before_connect_is_safe_noop(mocker: MockFixture):
    """``disconnect()`` is safe before ``connect()``. With the httpx client
    lazy-constructed (no work done in ``__init__``), there is nothing to close
    on a never-used client, so ``aclose`` should not be invoked."""
    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    closed = []

    async def fake_aclose(self):
        closed.append(True)

    mocker.patch("httpx.AsyncClient.aclose", new=fake_aclose)

    await client.disconnect()

    # Nothing was ever instantiated, so nothing to close.
    assert closed == []
    assert client._read_task is None
    assert client._write_task is None


async def test_construct_only_does_not_instantiate_httpx_client(
    mocker: MockFixture,
):
    """Constructing an ``AsyncStreamingClient`` and never calling
    ``connect()`` / ``create_temporary_token()`` / ``disconnect()`` must not
    instantiate an ``httpx.AsyncClient`` — otherwise an unused client leaks
    the pool. The HTTP client should be built lazily on first use."""
    import httpx

    constructed = []
    real_init = httpx.AsyncClient.__init__

    def counting_init(self, *args, **kwargs):
        constructed.append(True)
        return real_init(self, *args, **kwargs)

    mocker.patch.object(httpx.AsyncClient, "__init__", counting_init)

    AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )

    assert constructed == [], (
        "AsyncStreamingClient should not eagerly instantiate httpx.AsyncClient; "
        "got constructions: " + str(constructed)
    )


async def test_async_context_manager_calls_disconnect_on_exit(mocker: MockFixture):
    """``async with AsyncStreamingClient(opts) as c:`` must invoke
    ``disconnect()`` on block exit so callers can't forget cleanup."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    async with AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    ) as client:
        await client.connect(_default_params())
        await client.stream(b"\x00" * 32)

    # On exit, disconnect should have torn down read/write tasks.
    assert client._read_task is not None and client._read_task.done()
    assert client._write_task is not None and client._write_task.done()
    assert client._stop_event is not None and client._stop_event.is_set()


async def test_async_context_manager_disconnect_runs_on_exception(
    mocker: MockFixture,
):
    """Exception inside the ``async with`` body must still trigger
    ``disconnect()`` so the websocket / http client don't leak when user
    code raises."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    class _Boom(Exception):
        pass

    client_ref = {}

    with pytest.raises(_Boom):
        async with AsyncStreamingClient(
            StreamingClientOptions(api_key="test", api_host="api.example.com")
        ) as client:
            client_ref["c"] = client
            await client.connect(_default_params())
            raise _Boom()

    client = client_ref["c"]
    assert client._stop_event is not None and client._stop_event.is_set()
    assert client._websocket is None or fake_ws.close_call_count >= 1


async def test_disconnect_closes_http_client_when_used(mocker: MockFixture):
    """Once the lazy ``httpx.AsyncClient`` has been instantiated (by a call
    that goes through HTTP — e.g. ``create_temporary_token``), ``disconnect``
    must close it so the pool doesn't leak."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    async def fake_get(self, url, params=None):
        class _R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"token": "t"}

        return _R()

    mocker.patch("httpx.AsyncClient.get", new=fake_get)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())
    # Force the http client to be instantiated.
    await client.create_temporary_token(expires_in_seconds=60)

    closed = []

    async def fake_aclose(self):
        closed.append(True)

    mocker.patch("httpx.AsyncClient.aclose", new=fake_aclose)

    await client.disconnect()

    assert closed == [True]


async def test_server_error_dedups_concurrent_write_side_close(mocker: MockFixture):
    """Regression: a slow async ``on_error`` handler must not race a concurrent
    write-side ``ConnectionClosed`` into a duplicate dispatch. The
    ``_server_error_reported`` flag is set synchronously before the first
    ``await`` in ``_report_server_error`` — this test locks in that ordering."""
    close_exc = ConnectionClosed(rcvd=Close(1011, "send-side close"), sent=None)
    fake_ws = _FakeAsyncWebSocket(send_raises=close_exc)
    _patch_connect(mocker, fake_ws)

    received = []
    handler_started = asyncio.Event()
    handler_release = asyncio.Event()

    async def slow_on_error(_client, err):
        received.append(err)
        handler_started.set()
        await handler_release.wait()

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, slow_on_error)
    await client.connect(_default_params())

    # Push a server Error frame; the read task enters the slow handler.
    fake_ws.push_message(
        json.dumps({"type": "Error", "error": "boom", "error_code": 4002})
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1.0)

    # While the handler is parked, trigger a write-side close concurrently.
    await client.stream(b"\x00" * 32)
    for _ in range(50):
        if fake_ws.send_call_count >= 1:
            break
        await asyncio.sleep(0.01)

    # Release the handler; the read task finishes dispatch and exits.
    handler_release.set()

    await _wait_for_tasks(client)

    assert len(received) == 1, (
        f"expected exactly one on_error despite concurrent write-side close, "
        f"got {received}"
    )
    assert received[0].code == 4002

    await client.disconnect()


async def test_disconnect_during_slow_handler_tears_down(mocker: MockFixture):
    """Regression: ``disconnect()`` while an async handler is parked in a long
    ``await`` must cleanly cancel the read task. ``CancelledError`` is a
    ``BaseException`` (not ``Exception``), so it propagates through
    ``_invoke_handler`` and out of the read task — ``disconnect()`` then
    completes the cleanup."""
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    handler_started = asyncio.Event()

    async def slow_handler(_client, _event):
        handler_started.set()
        await asyncio.sleep(60)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Begin, slow_handler)
    await client.connect(_default_params())

    fake_ws.push_message(
        json.dumps({"type": "Begin", "id": "abc", "expires_at": "2030-01-01T00:00:00"})
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1.0)

    # disconnect() should not hang waiting for the parked sleep — the read
    # task is cancelled, CancelledError propagates, and disconnect returns.
    await asyncio.wait_for(client.disconnect(), timeout=2.0)

    assert client._read_task.done()


async def test_write_side_close_is_dispatched_when_read_short_circuits_on_stop(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    """Regression: if the read task observes ``_stop_event`` at the top of its
    loop (e.g. after processing a buffered message) before its next ``recv()``
    raises, the write task must still dispatch the connection-closed event.
    Previously the write task only set stop and exited, so this close went
    unreported."""
    caplog.set_level(logging.ERROR)

    close_exc = ConnectionClosed(rcvd=Close(1011, "send-side close"), sent=None)
    fake_ws = _FakeAsyncWebSocket(send_raises=close_exc)
    _patch_connect(mocker, fake_ws)

    received = []

    def on_error(_client, err):
        received.append(err)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)
    await client.connect(_default_params())

    # Queue a write so the write task hits send() and raises ConnectionClosed.
    await client.stream(b"\x00" * 32)

    # Wait for write task to finish dispatching the close.
    for _ in range(200):
        if received:
            break
        await asyncio.sleep(0.01)

    assert len(received) == 1, (
        f"expected exactly one on_error from write-side close, got {received}"
    )
    assert received[0].code == 1011

    await client.disconnect()
