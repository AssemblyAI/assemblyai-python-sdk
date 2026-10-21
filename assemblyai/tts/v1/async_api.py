from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from .api import (
    ENDPOINT_SPEECH,
    _cut_error,
    _error_from_response,
    _header_fields,
    _info_from_headers,
)
from .models import TTSResponse, TTSResponseInfo


class AsyncTTSStream:
    """
    The audio of one request, read as the API generates it.

    The asyncio counterpart of `TTSStream`, returned by
    `AsyncTextToSpeech.stream()`. The request is sent on `async with` or on
    the first read, and `TTSError` is raised there if the API refuses it.
    Use `async with`, or `await aclose()`, so the connection is released (a
    fully read stream's connection returns to the pool, a partially read one
    is dropped). `info` holds what the response headers say once the request
    has been sent; `aiter_bytes()` yields the audio as it arrives and
    `aread()` collects all of it. A failure after the first audio cuts the
    body: the iteration then raises `TTSError` with `retryable=True`.
    """

    def __init__(self, client: httpx.AsyncClient, request: httpx.Request) -> None:
        self._client = client
        self._request = request
        self._response: Optional[httpx.Response] = None
        self._info: Optional[TTSResponseInfo] = None

    @property
    def info(self) -> Optional[TTSResponseInfo]:
        """The response headers; `None` until the request has been sent."""

        return self._info

    async def _open(self) -> httpx.Response:
        if self._response is not None:
            return self._response

        response = await self._client.send(self._request, stream=True)
        if response.status_code != httpx.codes.OK:
            try:
                await response.aread()
                raise _error_from_response(response)
            finally:
                await response.aclose()

        self._response = response
        self._info = _info_from_headers(response.headers)
        return response

    async def __aenter__(self) -> "AsyncTTSStream":
        await self._open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.aiter_bytes()

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """Yields the audio bytes as they arrive."""

        response = await self._open()
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        except httpx.RemoteProtocolError as exc:
            raise _cut_error(self._info) from exc

    async def aread(self) -> bytes:
        """Reads the rest of the audio and returns it."""

        chunks = []
        async for chunk in self.aiter_bytes():
            chunks.append(chunk)
        return b"".join(chunks)

    async def aclose(self) -> None:
        """Releases the connection. Audio not yet read is discarded."""

        if self._response is not None:
            await self._response.aclose()


async def synthesize(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: Union[float, httpx.Timeout],
) -> TTSResponse:
    """
    Posts a buffered speech request and returns the complete audio.

    The asyncio counterpart of `api.synthesize`.

    Raises: `TTSError` on any non-200 response.
    """

    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_SPEECH,
        json={**body, "stream": False},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return TTSResponse(audio=response.content, **_header_fields(response.headers))


def stream_speech(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
) -> AsyncTTSStream:
    """
    Prepares a streamed speech request and returns it as an `AsyncTTSStream`.

    The asyncio counterpart of `api.stream_speech`: nothing is sent here.
    """

    request = client.build_request(
        "POST",
        base_url.rstrip("/") + ENDPOINT_SPEECH,
        json={**body, "stream": True},
        timeout=timeout,
    )
    return AsyncTTSStream(client, request)
