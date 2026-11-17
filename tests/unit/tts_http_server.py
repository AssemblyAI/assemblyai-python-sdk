"""A local HTTP/1.1 Text-to-Speech host for the connection tests of
`test_tts.py` and `test_tts_async.py`: it answers every POST with audio and
counts the TCP connections it accepts, so a test can tell a reused
connection from a new one."""

import contextlib
import http.server
import threading
import time
from typing import Dict, Iterator, Optional, cast


class CountingTTSServer(http.server.ThreadingHTTPServer):
    """
    Answers every POST with `audio` and `headers`, and keeps the connection
    open for the next request (HTTP/1.1 keep-alive).

    With `close_after`, the server closes each connection that many seconds
    after its response, as a load balancer or NAT closes an idle connection
    while the client still holds it.
    """

    daemon_threads = True

    def __init__(
        self,
        audio: bytes,
        headers: Dict[str, str],
        close_after: Optional[float] = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _AudioHandler)
        self.audio = audio
        self.response_headers = headers
        self.close_after = close_after
        self.connections = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def get_request(self):  # type: ignore[no-untyped-def]
        request = super().get_request()
        self.connections += 1
        return request


class _AudioHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keeps connections open between requests

    def do_POST(self) -> None:
        server = cast(CountingTTSServer, self.server)
        self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        for name, value in server.response_headers.items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(server.audio)))
        self.end_headers()
        self.wfile.write(server.audio)
        if server.close_after is not None:
            # The response carries no `Connection: close`, so the client keeps
            # the connection in its pool; the server then closes it.
            time.sleep(server.close_after)
            self.close_connection = True

    def log_message(self, *args: object) -> None:
        pass


@contextlib.contextmanager
def serving(
    audio: bytes,
    headers: Dict[str, str],
    close_after: Optional[float] = None,
) -> Iterator[CountingTTSServer]:
    """Runs a `CountingTTSServer` on a background thread for the block."""

    server = CountingTTSServer(audio, headers, close_after)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
