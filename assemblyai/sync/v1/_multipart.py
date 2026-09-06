"""Multipart encoding for streamed uploads.

Kept apart from `_base.py` so the transport modules can use it: `_base.py`
imports `api.py`, so anything `api.py` needs must not live there.
"""

from __future__ import annotations

import json
import secrets
from typing import (
    AsyncIterable,
    AsyncIterator,
    BinaryIO,
    Iterable,
    Iterator,
    Optional,
    Union,
)

# Audio for a streamed upload, delivered in whatever pieces the producer has
# ready. A file object is read lazily. Audio the caller already holds whole
# belongs in `transcribe()`, which is faster for it.
AudioChunks = Union[Iterable[bytes], BinaryIO]
AsyncAudioChunks = Union[AsyncIterable[bytes], Iterable[bytes], BinaryIO]

# Read size for file objects handed to the streaming path.
_STREAM_READ_SIZE = 32 * 1024


class StreamingMultipartEncoder:
    """
    Builds a `multipart/form-data` body one part at a time.

    The buffered path hands httpx a `files=` mapping and lets it encode the
    whole body up front. A streamed upload cannot: the audio does not exist yet
    when the request starts. This emits the framing itself so the `config` part
    can go out immediately and audio can follow as it arrives.

    **Part order is significant and is enforced here by construction.** The
    server decodes audio as it lands, so it needs `sample_rate` and `channels`
    before the first audio byte; `config` therefore precedes `audio`. Audio
    first is rejected by the server rather than buffered, which is the whole
    point of the endpoint.
    """

    def __init__(self, boundary: Optional[str] = None) -> None:
        """
        Args:
            boundary: the multipart boundary. Generated when absent; pass one
                only to make a test deterministic.
        """
        self.boundary = boundary or secrets.token_hex(16)

    @property
    def content_type(self) -> str:
        """The `Content-Type` header the body must be sent with."""

        return f"multipart/form-data; boundary={self.boundary}"

    def config_part(self, config: Optional[dict]) -> bytes:
        """Encodes the JSON `config` part. Empty when there is no config."""

        if not config:
            return b""

        return (
            (
                f"--{self.boundary}\r\n"
                'Content-Disposition: form-data; name="config"\r\n'
                "Content-Type: application/json\r\n\r\n"
            ).encode("utf-8")
            + json.dumps(config).encode("utf-8")
            + b"\r\n"
        )

    def audio_header(self, filename: str, content_type: str) -> bytes:
        """Encodes the `audio` part's headers, which precede its first byte."""

        return (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")

    def closing(self) -> bytes:
        """Encodes the terminating boundary."""

        return f"\r\n--{self.boundary}--\r\n".encode("utf-8")


def iter_chunks(data: AudioChunks) -> Iterator[bytes]:
    """Yields audio pieces from a file object or any iterable of bytes."""

    if hasattr(data, "read"):
        while True:
            chunk = data.read(_STREAM_READ_SIZE)
            if not chunk:
                return
            yield chunk
    else:
        for chunk in data:  # type: ignore[union-attr]
            yield chunk


async def aiter_chunks(data: AsyncAudioChunks) -> AsyncIterator[bytes]:
    """
    Yields audio pieces from an async iterable, a file object, or a plain
    iterable.

    A synchronous source is accepted so one producer can feed both
    transcribers, but it is consumed inline: a blocking `read()` will block the
    event loop.
    """
    if hasattr(data, "__aiter__"):
        async for chunk in data:  # type: ignore[union-attr]
            yield chunk
    else:
        for chunk in iter_chunks(data):  # type: ignore[arg-type]
            yield chunk
