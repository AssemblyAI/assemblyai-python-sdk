"""The asyncio counterpart of `client.py`."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Callable, Optional, Type, TypeVar

import httpx
from typing_extensions import Self

from ... import async_client as _async_client
from ... import types
from . import api, async_api
from ._base import AudioInput, _config_to_json, _resolve_audio

_T = TypeVar("_T")


async def _run_in_thread(func: Callable[..., _T], *args: Any) -> _T:
    """Runs a blocking call on the default executor."""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None, func, *args)


class AsyncSyncTranscriber:
    """
    The asyncio counterpart of `SyncTranscriber`: audio in, transcript out,
    one request — without blocking the event loop.

    Like `SyncTranscriber`, it posts the audio to the sync API and returns
    the finished `SyncTranscriptResponse` directly; there is no job id or
    status to poll. Accepts a local file path, raw bytes, or a binary file
    object — but not a URL. Use it in asyncio code (FastAPI, aiohttp, voice
    agents), where `SyncTranscriber.transcribe()` would block the loop and
    `transcribe_async()`'s `concurrent.futures.Future` is not awaitable.

    The transcriber owns an HTTP connection pool. Close it with `aclose()`,
    or use the transcriber as an async context manager.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncSyncTranscriber() as transcriber:
                result = await transcriber.transcribe("./call.wav")
                print(result.text)

        asyncio.run(main())
        ```

        Transcribing several clips concurrently is plain asyncio:
        ```python
        async with aai.AsyncSyncTranscriber() as transcriber:
            results = await asyncio.gather(
                transcriber.transcribe("./one.wav"),
                transcriber.transcribe("./two.wav"),
            )
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> None:
        """
        Creates an `AsyncSyncTranscriber`.

        Args:
            client: The `AsyncClient` to use. If `None`, the transcriber
                creates one from the global `settings` and closes it on
                `aclose()`. Pass a client to share one pool between
                transcribers.
            config: Default transcription options. Per-call `config`
                overrides it.
        """
        from ... import settings as default_settings

        self._owns_client = client is None
        self._client = client or _async_client.AsyncClient(settings=default_settings)
        self.config = config or types.SyncTranscriptionConfig()

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this transcriber sends requests with."""

        return self._client

    async def transcribe(
        self,
        data: AudioInput,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> types.SyncTranscriptResponse:
        """
        Transcribes audio and returns the finished transcript.

        Reads path and file-object input off the event loop.

        Args:
            data: A local file path, raw audio bytes, or a binary file object.
                Raw PCM also requires `sample_rate` and `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises: `SyncTranscriptError` if the request fails.
        """
        config = config or self.config
        audio, filename, content_type = await _run_in_thread(
            _resolve_audio, data, config
        )

        return await async_api.transcribe(
            self._client.http_client,
            base_url=self._client.settings.sync_base_url,
            audio=audio,
            filename=filename,
            audio_content_type=content_type,
            model=config.model,
            config=_config_to_json(config),
            timeout=self._client.settings.sync_http_timeout,
        )

    async def warm(self) -> bool:
        """
        Opens the connection to the sync API ahead of time.

        The sync API is a single request/response, so a `transcribe()` that
        opens its connection on demand pays the full DNS + TCP + TLS handshake
        on the critical path — one network round trip that, for a distant
        client, can rival the transcription itself. Awaiting `warm()` as soon
        as you know audio is coming — typically while the clip is still being
        recorded, e.g. via `asyncio.create_task(transcriber.warm())` — spends
        that setup concurrently: the next `transcribe()` reuses the
        already-open connection.

        The warmed connection is reused while it stays in the HTTP pool —
        `settings.keepalive_expiry` seconds (httpx's 5s default unless raised).
        Call `warm()` shortly before `transcribe()`, or raise
        `keepalive_expiry` (e.g. to 120, the sync audio cap) so a single call
        covers a whole in-progress recording. `warm()` is idempotent and cheap,
        so calling it again to refresh the connection is fine.

        Routing the same `config.model` as the eventual transcription ensures
        the warmed connection lands on the right backend.

        Returns:
            True once the connection is open (any HTTP response — even a
            non-200 — means the socket is established); False if the
            connection could not be opened (transport error).
        """
        settings = self._client.settings
        url = settings.sync_base_url.rstrip("/") + api.ENDPOINT_WARM
        try:
            await self._client.http_client.get(
                url,
                headers={api.MODEL_HEADER: self.config.model},
                timeout=min(settings.sync_http_timeout, 10.0),
            )
        except httpx.HTTPError:
            return False
        return True

    async def aclose(self) -> None:
        """
        Closes the HTTP connection pool.

        Leaves a client that was passed in alone. Its creator closes it.
        """

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.aclose()
