import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from websockets.exceptions import InvalidStatus

from assemblyai.streaming.v3 import (
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
)


class Socket:
    def __init__(self):
        self.incoming = queue.Queue()
        self.incoming.put(
            json.dumps(
                {
                    "type": "Begin",
                    "id": "offline-session",
                    "expires_at": "2030-01-01T00:00:00Z",
                }
            )
        )
        self.sent = []
        self.audio_sent = threading.Event()
        self.closed = False

    def recv(self, timeout=1):
        try:
            return self.incoming.get(timeout=min(timeout, 0.05))
        except queue.Empty:
            raise TimeoutError

    def send(self, data):
        assert not self.closed
        self.sent.append(data)
        if isinstance(data, bytes):
            self.audio_sent.set()
        elif json.loads(data)["type"] == "Terminate":
            self.incoming.put(json.dumps({"type": "Termination"}))

    def close(self):
        self.closed = True


@pytest.fixture
def client():
    transcriber = RealTimeTranscriber(
        RealTimeTranscriberOptions(
            api_key="offline-test",
            max_connection_retries=0,
            connection_retry_delay=0,
        )
    )
    yield transcriber
    transcriber.disconnect()
    transcriber._client._http_client.close()
    assert not transcriber._read_thread.is_alive()
    assert not transcriber._write_thread.is_alive()


def params():
    return RealTimeParameters(sample_rate=16000, speech_model="universal-3-5-pro")


@pytest.mark.parametrize("disconnect_first", [False, True])
def test_repeated_connect_does_not_allocate_socket(client, mocker, disconnect_first):
    socket = Socket()
    connect = mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect", return_value=socket
    )
    begun = threading.Event()
    client.on(RealTimeEvents.Begin, lambda c, e: begun.set())
    client.connect(params())
    assert begun.wait(2)
    client.stream(b"\0" * 3200)
    assert socket.audio_sent.wait(2)
    assert socket.sent == [b"\0" * 3200]
    if disconnect_first:
        client.disconnect()
        assert socket.closed
    with pytest.raises(RuntimeError) as error:
        client.connect(params())
    assert connect.call_count == 1
    assert "create a new instance" in str(error.value)
    client.disconnect()
    assert socket.closed


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("offline timeout"), InvalidStatus(SimpleNamespace(status_code=401))],
)
def test_failed_handshake_consumes_client(client, mocker, failure):
    connect = mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect", side_effect=failure
    )
    errors = []
    client.on(RealTimeEvents.Error, lambda c, e: errors.append(e))
    client.connect(params())
    assert len(errors) == 1
    with pytest.raises(RuntimeError, match="create a new instance"):
        client.connect(params())
    assert connect.call_count == 1


def test_concurrent_connect_reserves_before_handshake(client, mocker):
    entered = threading.Event()
    release = threading.Event()
    socket = Socket()

    def handshake(*args, **kwargs):
        assert not entered.is_set(), "unexpected second handshake"
        entered.set()
        assert release.wait(2)
        return socket

    connect = mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect", side_effect=handshake
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(client.connect, params())
        try:
            assert entered.wait(2)
            with pytest.raises(RuntimeError, match="create a new instance"):
                client.connect(params())
        finally:
            release.set()
            first.result(timeout=3)
    assert connect.call_count == 1
    client.disconnect()
    assert socket.closed


def test_internal_handshake_retry_still_streams(client, mocker):
    client._options.max_connection_retries = 1
    socket = Socket()
    connect = mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        side_effect=[TimeoutError("offline timeout"), socket],
    )
    terminated = threading.Event()
    client.on(RealTimeEvents.Termination, lambda c, e: terminated.set())
    client.connect(params())
    client.stream(b"\0" * 3200)
    assert socket.audio_sent.wait(2)
    client.disconnect(terminate=True)
    assert terminated.is_set()
    assert socket.closed
    assert connect.call_count == 2


@pytest.mark.parametrize("thread_name", ["_read_thread", "_write_thread"])
def test_thread_start_failure_closes_owned_socket(client, mocker, thread_name):
    socket = Socket()
    connect = mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect", return_value=socket
    )
    mocker.patch.object(
        getattr(client, thread_name), "start", side_effect=RuntimeError("cannot start")
    )
    with pytest.raises(RuntimeError, match="cannot start"):
        client.connect(params())
    assert socket.closed
    with pytest.raises(RuntimeError, match="create a new instance"):
        client.connect(params())
    assert connect.call_count == 1
