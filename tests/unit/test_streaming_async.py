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
    assert len(close_logs) == 1

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

    async def fake_get(self, url, params=None):
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

    # Clean up the (un-mocked) AsyncClient so the test doesn't emit
    # "unclosed transport" warnings.
    await client._client.aclose()


async def test_connect_twice_raises(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    with pytest.raises(RuntimeError, match="already connected"):
        await client.connect(_default_params())

    await client.disconnect()


async def test_disconnect_closes_http_client(mocker: MockFixture):
    fake_ws = _FakeAsyncWebSocket()
    _patch_connect(mocker, fake_ws)

    client = AsyncStreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    await client.connect(_default_params())

    closed = []

    async def fake_aclose(self):
        closed.append(True)

    mocker.patch("httpx.AsyncClient.aclose", new=fake_aclose)

    await client.disconnect()

    assert closed == [True]


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

    assert (
        len(received) == 1
    ), f"expected exactly one on_error from write-side close, got {received}"
    assert received[0].code == 1011

    await client.disconnect()
