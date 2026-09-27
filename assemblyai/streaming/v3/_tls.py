"""TLS context for the thread-based streaming client.

websockets' sync client starts the thread that reads the connection before the
calling thread writes the opening handshake request. With servers that send
TLS 1.3 post-handshake messages, such as session tickets, a read already
blocked on the socket while that request is written occasionally leaves the
connection without a handshake response, and the connect times out. Sockets
created from ``_create_tls_context()`` hold reads until the first write has
returned, so the request is written before anything is read.

Reads are held only on sockets the context builds from its
``sslsocket_class``. An ``SSLContext`` replacement whose ``wrap_socket()``
builds sockets some other way keeps its own sockets, and reads on them are not
held.
"""

from __future__ import annotations

import ssl
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer, WriteableBuffer

# Upper bound on how long a read is held, so a socket that is never written to
# can still be read.
_FIRST_WRITE_WAIT_SEC = 30.0

# Serializes the creation of each socket's first-write event.
_FIRST_WRITE_LOCK = threading.Lock()


class _WriteFirstSSLSocket(ssl.SSLSocket):
    """``SSLSocket`` whose reads wait until the first write has returned.

    The first ``send()`` or ``sendall()`` releases held reads whether or not it
    succeeds, and so does ``shutdown()``, which lets closing the socket
    interrupt a held read. Once released, reads and writes proceed
    concurrently as with a plain ``SSLSocket``.
    """

    _first_write_event: Optional[threading.Event] = None

    def _first_write_done(self) -> threading.Event:
        # SSLContext.wrap_socket() builds instances without calling __init__,
        # so the event is created on first use.
        done = self._first_write_event
        if done is None:
            with _FIRST_WRITE_LOCK:
                done = self._first_write_event
                if done is None:
                    done = self._first_write_event = threading.Event()
        return done

    def send(self, data: ReadableBuffer, flags: int = 0) -> int:
        try:
            return super().send(data, flags)
        finally:
            self._first_write_done().set()

    def sendall(self, data: ReadableBuffer, flags: int = 0) -> None:
        try:
            super().sendall(data, flags)
        finally:
            self._first_write_done().set()

    def recv(self, buflen: int = 1024, flags: int = 0) -> bytes:
        self._first_write_done().wait(_FIRST_WRITE_WAIT_SEC)
        return super().recv(buflen, flags)

    def recv_into(
        self,
        buffer: WriteableBuffer,
        nbytes: Optional[int] = None,
        flags: int = 0,
    ) -> int:
        self._first_write_done().wait(_FIRST_WRITE_WAIT_SEC)
        return super().recv_into(buffer, nbytes, flags)

    def shutdown(self, how: int) -> None:
        self._first_write_done().set()
        super().shutdown(how)


def _create_tls_context() -> ssl.SSLContext:
    """Return a default client context whose sockets are
    ``_WriteFirstSSLSocket``.

    Certificate and hostname verification are those of
    ``ssl.create_default_context()``, which is also what websockets uses when
    no context is given.

    ``_WriteFirstSSLSocket`` extends the ``ssl.SSLSocket`` present when this
    module was imported. If the default context uses a socket class that
    ``_WriteFirstSSLSocket`` does not extend, for example because gevent
    patched ``ssl`` afterwards, the context keeps that class: its
    ``wrap_socket()`` may not be able to construct ``_WriteFirstSSLSocket``.
    """
    context = ssl.create_default_context()
    if issubclass(_WriteFirstSSLSocket, context.sslsocket_class):
        context.sslsocket_class = _WriteFirstSSLSocket
    return context
