from typing import Any, AsyncContextManager, AsyncIterator, Dict, Optional

import httpx

from .api import (
    _CUT_ERRORS,
    ENDPOINT_SPEECH,
    Timeout,
    _cut_error,
    _error_from_response,
    _header_fields,
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

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        body: Dict[str, Any],
        timeout: Timeout,
    ) -> None:
        self._client = client
        self._url = url
        self._body = body
        self._timeout = timeout
        self._cm: Optional[AsyncContextManager[httpx.Response]] = None
        self._response: Optional[httpx.Response] = None
        self._info: Optional[TTSResponseInfo] = None

    @property
    def info(self) -> TTSResponseInfo:
        """The response headers, once the stream is entered or first read."""

        if self._info is None:
            raise RuntimeError("enter or read the stream before `info`")
        return self._info

    async def _open(self) -> httpx.Response:
        if self._response is not None:
            return self._response

        cm = self._client.stream(
            "POST", self._url, json=self._body, timeout=self._timeout
        )
        response = await cm.__aenter__()
        if response.status_code != httpx.codes.OK:
            try:
                await response.aread()
                raise _error_from_response(response)
            finally:
                await cm.__aexit__(None, None, None)

        self._cm = cm
        self._response = response
        self._info = TTSResponseInfo(
            **_header_fields(response.headers, self._body.get("voice_id"))
        )
        return response

    async def __aenter__(self) -> "AsyncTTSStream":
        await self._open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.aiter_bytes()

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """
        Yields the audio bytes as they arrive.

        Raises:
            TTSError: from the first read if the API refuses the request, or
                later if the body is cut before the audio is complete.
        """

        response = await self._open()
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        except _CUT_ERRORS as error:
            raise _cut_error(self._info, error) from error

    async def aread(self) -> bytes:
        """Reads the rest of the audio and returns it."""

        chunks = []
        async for chunk in self.aiter_bytes():
            chunks.append(chunk)
        return b"".join(chunks)

    async def aclose(self) -> None:
        """Releases the connection. Audio not yet read is discarded."""

        cm, self._cm = self._cm, None
        if cm is not None:
            await cm.__aexit__(None, None, None)


async def synthesize(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: Timeout,
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

    return TTSResponse(
        audio=response.content,
        **_header_fields(response.headers, body.get("voice_id")),
    )


def stream_speech(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: Timeout,
) -> AsyncTTSStream:
    """
    Prepares a streamed speech request and returns it as an `AsyncTTSStream`.

    The asyncio counterpart of `api.stream_speech`: nothing is sent here.
    """

    return AsyncTTSStream(
        client,
        url=base_url.rstrip("/") + ENDPOINT_SPEECH,
        body={**body, "stream": True},
        timeout=timeout,
    )
