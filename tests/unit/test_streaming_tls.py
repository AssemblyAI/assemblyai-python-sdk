import contextlib
import inspect
import shutil
import socket
import ssl
import subprocess
import threading
import warnings
from typing import Iterator, List, Optional, Tuple

import pytest
import websockets
from pytest_mock import MockFixture
from websockets.sync.server import serve as websocket_serve

from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    _tls,
)
from assemblyai.streaming.v3.client import websocket_connect

# websockets 13 renamed the sync ``connect()`` / ``serve()`` TLS parameter
# from ``ssl_context`` to ``ssl``.
_WEBSOCKETS_MAJOR = int(websockets.__version__.split(".")[0])
_EXPECTED_SSL_KW = "ssl" if _WEBSOCKETS_MAJOR >= 13 else "ssl_context"
_SERVE_SSL_KW = (
    "ssl" if "ssl" in inspect.signature(websocket_serve).parameters else "ssl_context"
)

_HOST = "127.0.0.1"


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory: pytest.TempPathFactory) -> Tuple[str, str]:
    """A throwaway self-signed certificate and key for ``127.0.0.1``."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("the openssl CLI is needed to generate a test certificate")
    directory = tmp_path_factory.mktemp("tls")
    cert = str(directory / "cert.pem")
    key = str(directory / "key.pem")
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-nodes",
            "-days",
            "1",
            "-subj",
            f"/CN={_HOST}",
            "-addext",
            f"subjectAltName=IP:{_HOST}",
            "-keyout",
            key,
            "-out",
            cert,
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture
def server_context(tls_cert: Tuple[str, str]) -> ssl.SSLContext:
    """A server context that only speaks TLS 1.3."""
    cert, key = tls_cert
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(cert, key)
    return context


@pytest.fixture
def client_context(tls_cert: Tuple[str, str]) -> ssl.SSLContext:
    """The SDK context, additionally trusting the test certificate."""
    cert, _ = tls_cert
    context = _tls._create_tls_context()
    context.load_verify_locations(cert)
    return context


@contextlib.contextmanager
def _tls_server(context: ssl.SSLContext, greeting: bytes) -> Iterator[int]:
    """Accept one TLS connection, send ``greeting`` as soon as the handshake
    completes, then read until the client goes away. Yields the port."""
    listener = socket.create_server((_HOST, 0))
    listener.settimeout(10)

    def serve() -> None:
        with contextlib.suppress(OSError):
            conn, _ = listener.accept()
            with context.wrap_socket(conn, server_side=True) as tls:
                tls.sendall(greeting)
                while tls.recv(1024):
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()
        thread.join(timeout=10)


@contextlib.contextmanager
def _websocket_server(context: ssl.SSLContext) -> Iterator[int]:
    """Run a websockets echo server over TLS. Yields the port."""

    def echo(connection) -> None:
        for message in connection:
            connection.send(message)

    server = websocket_serve(echo, _HOST, 0, **{_SERVE_SSL_KW: context})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.socket.getsockname()[1]
    finally:
        server.shutdown()
        thread.join(timeout=10)


class _Read:
    """``recv()`` on a background thread, as websockets' reader thread does."""

    def __init__(self, sock: ssl.SSLSocket):
        self.done = threading.Event()
        self.result: Optional[bytes] = None
        self.error: Optional[BaseException] = None
        threading.Thread(target=self._run, args=(sock,), daemon=True).start()

    def _run(self, sock: ssl.SSLSocket) -> None:
        try:
            self.result = sock.recv(1024)
        except (OSError, ValueError) as exc:
            self.error = exc
        finally:
            self.done.set()


def _connect_tls(context: ssl.SSLContext, port: int) -> ssl.SSLSocket:
    return context.wrap_socket(
        socket.create_connection((_HOST, port), timeout=10),
        server_hostname=_HOST,
    )


class _OtherSSLSocket(ssl.SSLSocket):
    """A socket class that ``_WriteFirstSSLSocket`` does not extend."""


def test_tls_context_keeps_verification_on():
    context = _tls._create_tls_context()

    assert context.sslsocket_class is _tls._WriteFirstSSLSocket
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_tls_context_keeps_socket_class_it_does_not_extend(
    monkeypatch: pytest.MonkeyPatch,
):
    # Given: default contexts that build sockets of another class
    monkeypatch.setattr(ssl.SSLContext, "sslsocket_class", _OtherSSLSocket)

    # When: creating the SDK context
    context = _tls._create_tls_context()

    # Then: the context keeps that class and its verification settings
    assert context.sslsocket_class is _OtherSSLSocket
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_tls_context_rejects_untrusted_certificate(
    server_context: ssl.SSLContext,
):
    # Given: a server whose self-signed certificate the default trust store
    # does not contain
    with _tls_server(server_context, greeting=b"hello") as port:
        # When / Then: the handshake fails verification
        with pytest.raises(ssl.SSLCertVerificationError):
            _connect_tls(_tls._create_tls_context(), port)


def test_read_waits_for_first_write(
    client_context: ssl.SSLContext, server_context: ssl.SSLContext
):
    # Given: a TLS 1.3 connection whose server writes as soon as the handshake
    # completes
    with _tls_server(server_context, greeting=b"hello") as port:
        with _connect_tls(client_context, port) as sock:
            assert isinstance(sock, _tls._WriteFirstSSLSocket)
            assert sock.version() == "TLSv1.3"

            # When: a read starts before anything is written
            read = _Read(sock)

            # Then: it is held although the server's data is available
            assert not read.done.wait(0.5)

            # When: the first write returns
            sock.sendall(b"ping")

            # Then: the read proceeds and returns the server's data
            assert read.done.wait(10)
            assert read.error is None
            assert read.result == b"hello"


def test_shutdown_releases_held_read(
    client_context: ssl.SSLContext, server_context: ssl.SSLContext
):
    # Given: a read held because nothing has been written
    with _tls_server(server_context, greeting=b"hello") as port:
        with _connect_tls(client_context, port) as sock:
            read = _Read(sock)
            assert not read.done.wait(0.2)

            # When: the socket is shut down without ever being written to
            sock.shutdown(socket.SHUT_RDWR)

            # Then: the read returns well before the wait bound
            assert read.done.wait(5)


def test_read_proceeds_after_wait_bound(
    monkeypatch: pytest.MonkeyPatch,
    client_context: ssl.SSLContext,
    server_context: ssl.SSLContext,
):
    # Given: a short wait bound
    monkeypatch.setattr(_tls, "_FIRST_WRITE_WAIT_SEC", 0.2)

    with _tls_server(server_context, greeting=b"hello") as port:
        with _connect_tls(client_context, port) as sock:
            # When: a read starts and nothing is ever written
            read = _Read(sock)

            # Then: the read returns the server's data once the bound elapses
            assert read.done.wait(10)
            assert read.error is None
            assert read.result == b"hello"


def test_websocket_handshake_over_wss(
    monkeypatch: pytest.MonkeyPatch,
    tls_cert: Tuple[str, str],
    server_context: ssl.SSLContext,
):
    # Given: the SDK context, trusting the test certificate, and a TLS 1.3
    # websockets echo server
    cert, _ = tls_cert
    create_tls_context = _tls._create_tls_context

    def trusting_tls_context() -> ssl.SSLContext:
        context = create_tls_context()
        context.load_verify_locations(cert)
        return context

    monkeypatch.setattr(
        "assemblyai.streaming.v3.client._create_tls_context", trusting_tls_context
    )

    with _websocket_server(server_context) as port:
        # When: connecting over wss:// with the sync client's connect
        with warnings.catch_warnings():
            # The TLS parameter name matches the installed websockets, so no
            # rename DeprecationWarning is raised.
            warnings.simplefilter("error", DeprecationWarning)
            connection = websocket_connect(f"wss://{_HOST}:{port}/", open_timeout=10)

        # Then: the opening handshake completes over the write-first socket
        with connection:
            assert isinstance(connection.socket, _tls._WriteFirstSSLSocket)
            assert connection.socket.version() == "TLSv1.3"
            connection.send("ping")
            assert connection.recv(timeout=10) == "ping"


def _record_connect(mocker: MockFixture) -> List[Tuple[str, dict]]:
    calls: List[Tuple[str, dict]] = []

    def fake_connect(uri: str, **kwargs) -> None:
        calls.append((uri, kwargs))

    mocker.patch("assemblyai.streaming.v3.client._ws_sync_connect", new=fake_connect)
    mocker.patch("threading.Thread.start", return_value=None)
    return calls


def _connect_client(api_host: str) -> None:
    client = StreamingClient(StreamingClientOptions(api_key="test", api_host=api_host))
    client.connect(
        StreamingParameters(
            sample_rate=16000,
            speech_model=SpeechModel.universal_streaming_english,
        )
    )


def test_client_connect_passes_tls_context_for_wss(mocker: MockFixture):
    # Given: a client pointed at a wss:// host
    calls = _record_connect(mocker)

    # When: connecting
    _connect_client("api.example.com")

    # Then: websockets gets the SDK context under the TLS parameter name of
    # the installed version, alongside the usual arguments
    [(uri, kwargs)] = calls
    assert uri.startswith("wss://api.example.com/v3/ws?")
    assert {"ssl", "ssl_context"} & kwargs.keys() == {_EXPECTED_SSL_KW}
    context = kwargs[_EXPECTED_SSL_KW]
    assert isinstance(context, ssl.SSLContext)
    assert context.sslsocket_class is _tls._WriteFirstSSLSocket
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert kwargs["additional_headers"]["Authorization"] == "test"
    assert kwargs["open_timeout"] == 1.0


def test_client_connect_passes_no_tls_context_for_ws(mocker: MockFixture):
    # Given: a client pointed at a plain ws:// host
    calls = _record_connect(mocker)

    # When: connecting
    _connect_client("ws://localhost:8080")

    # Then: no TLS argument is passed, which websockets rejects for ws://
    [(uri, kwargs)] = calls
    assert uri.startswith("ws://localhost:8080/v3/ws?")
    assert not {"ssl", "ssl_context"} & kwargs.keys()
