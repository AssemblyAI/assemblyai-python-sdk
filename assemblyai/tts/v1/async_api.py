from typing import Any, AsyncIterator, Dict, Union

import httpx

from .api import (
    _CUT_ERRORS,
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
    `AsyncTextToSpeech.stream()`. Use `async with`, or `await aclose()`, so
    the connection is released (a fully read stream's connection returns to
    the pool, a partially read one is dropped). `info` holds what the response
    headers say; `aiter_bytes()` yields the audio as it arrives and `aread()`
    collects all of it. A failure after the first audio cuts the body: the
    iteration raises a `TTSError` with `status_code` `200`, `retryable`
    `True` and `info.request_id`, from the httpx error.
    """

    def __init__(self, response: httpx.Response, info: TTSResponseInfo) -> None:
        self._response = response
        self.info = info

    async def __aenter__(self) -> "AsyncTTSStream":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.aiter_bytes()

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """
        Yields the audio bytes as they arrive.

        Raises:
            TTSError: if the body is cut before the audio is complete.
        """

        try:
            async for chunk in self._response.aiter_bytes():
                if chunk:
                    yield chunk
        except _CUT_ERRORS as error:
            raise _cut_error(self.info, error) from error

    async def aread(self) -> bytes:
        """Reads the rest of the audio and returns it."""

        chunks = []
        async for chunk in self.aiter_bytes():
            chunks.append(chunk)
        return b"".join(chunks)

    async def aclose(self) -> None:
        """Releases the connection. Audio not yet read is discarded."""

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


async def stream_speech(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
) -> AsyncTTSStream:
    """
    Posts a streamed speech request and returns the open response.

    The asyncio counterpart of `api.stream_speech`.

    Raises: `TTSError` if the request fails before any audio is sent; the
    returned stream raises it if the body is cut after the first audio.
    """

    request = client.build_request(
        "POST",
        base_url.rstrip("/") + ENDPOINT_SPEECH,
        json={**body, "stream": True},
        timeout=timeout,
    )
    response = await client.send(request, stream=True)

    if response.status_code != httpx.codes.OK:
        try:
            await response.aread()
            raise _error_from_response(response)
        finally:
            await response.aclose()

    return AsyncTTSStream(response, _info_from_headers(response.headers))
