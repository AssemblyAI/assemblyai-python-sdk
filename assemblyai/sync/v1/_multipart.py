"""Multipart encoding for streamed uploads.

Kept apart from `_base.py` so the transport modules can use it: `_base.py`
imports `api.py`, so anything `api.py` needs must not live there.
"""

from __future__ import annotations

import asyncio
import json
import re
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

# Escaping for quoted `Content-Disposition` parameters, matching what httpx's
# own multipart encoder does (the HTML5 form-encoding rules): a double quote
# becomes `%22`, a backslash is doubled, and control characters other than ESC
# are percent-encoded so a name cannot break out of the header.
_FORM_PARAM_REPLACEMENTS = {'"': "%22", "\\": "\\\\"}
_FORM_PARAM_REPLACEMENTS.update(
    {chr(c): "%{:02X}".format(c) for c in range(0x1F + 1) if c != 0x1B}
)
_FORM_PARAM_RE = re.compile("|".join(map(re.escape, _FORM_PARAM_REPLACEMENTS)))


def _escape_form_param(value: str) -> str:
    """Escapes `value` for use inside a quoted multipart header parameter."""

    return _FORM_PARAM_RE.sub(lambda m: _FORM_PARAM_REPLACEMENTS[m.group(0)], value)


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
        """
        Encodes the JSON `config` part.

        Always emitted: the streaming endpoint rejects a body whose audio is
        not preceded by a `config` part, so when the caller set no options an
        empty object is sent. The buffered route omits the part instead.
        """

        return (
            (
                f"--{self.boundary}\r\n"
                'Content-Disposition: form-data; name="config"\r\n'
                "Content-Type: application/json\r\n\r\n"
            ).encode("utf-8")
            + json.dumps(config or {}).encode("utf-8")
            + b"\r\n"
        )

    def audio_header(self, filename: str, content_type: str) -> bytes:
        """
        Encodes the `audio` part's headers, which precede its first byte.

        `filename` is escaped so quotes and control characters in a file's
        name cannot corrupt the framing.
        """

        return (
            f"--{self.boundary}\r\n"
            "Content-Disposition: form-data; "
            f'name="audio"; filename="{_escape_form_param(filename)}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")

    def closing(self) -> bytes:
        """Encodes the terminating boundary."""

        return f"\r\n--{self.boundary}--\r\n".encode("utf-8")


def _as_bytes(chunk: object) -> bytes:
    """
    Returns `chunk` as `bytes`, or raises if it is not binary audio.

    Names the likely cause when the producer yields text — a file opened
    without `"b"`, or a generator of `str` — so the mistake does not surface
    as an opaque failure deep inside the transport.
    """

    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, (bytearray, memoryview)):
        return bytes(chunk)
    if isinstance(chunk, str):
        raise TypeError(
            "transcribe_stream() audio chunks must be bytes, not str. Open the "
            "file in binary mode ('rb') or encode the producer's output."
        )
    raise TypeError(
        f"transcribe_stream() audio chunks must be bytes, not {type(chunk).__name__}"
    )


def iter_chunks(data: AudioChunks) -> Iterator[bytes]:
    """
    Yields audio pieces from a file object or any iterable of bytes.

    Raises `TypeError` on the first piece that is not bytes-like.
    """

    if hasattr(data, "read"):
        while True:
            chunk = data.read(_STREAM_READ_SIZE)
            if not chunk:
                return
            yield _as_bytes(chunk)
    else:
        for chunk in data:  # type: ignore[union-attr]
            yield _as_bytes(chunk)


async def aiter_chunks(data: AsyncAudioChunks) -> AsyncIterator[bytes]:
    """
    Yields audio pieces from an async iterable, a file object, or a plain
    iterable.

    A file object is read in a worker thread, so a `read()` that blocks on a
    pipe or socket does not stall the event loop. A plain iterable is accepted
    so one producer can feed both transcribers, but it is consumed inline and
    so must not block.

    Raises `TypeError` on the first piece that is not bytes-like.
    """
    if hasattr(data, "__aiter__"):
        async for chunk in data:  # type: ignore[union-attr]
            yield _as_bytes(chunk)
    elif hasattr(data, "read"):
        while True:
            chunk = await asyncio.to_thread(data.read, _STREAM_READ_SIZE)  # type: ignore[union-attr]
            if not chunk:
                return
            yield _as_bytes(chunk)
    else:
        for chunk in data:  # type: ignore[union-attr]
            yield _as_bytes(chunk)
