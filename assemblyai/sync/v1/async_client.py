"""The asyncio counterpart of `client.py`."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, AsyncIterator, Callable, Optional, Type, TypeVar

import httpx
from typing_extensions import Self

from ... import async_client as _async_client
from ... import types
from . import api, async_api
from ._base import (
    AudioInput,
    _Aborted,
    _config_to_json,
    _resolve_audio,
    check_chunks,
    check_config,
    stream_filename,
)
from ._multipart import AsyncAudioChunks, _as_bytes

_T = TypeVar("_T")


async def _run_in_thread(func: Callable[..., _T], *args: Any) -> _T:
    """Runs a blocking call on the default executor."""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None, func, *args)


class AsyncLiveSession:
    """
    A live upload that audio is pushed into, for asyncio code.

    Returned by `AsyncSyncTranscriber.open_live()`. The request runs as a task
    on the current event loop from the moment the session opens; `write()`
    hands it audio, `close()` ends the audio, and `await result()` waits for
    the transcript. Built for callback-driven sources — a WebRTC track, a
    websocket handler receiving frames, a telephony media stream — where the
    audio arrives in a callback rather than from an async iterator you can
    hand to `transcribe_live()`.

    `write()` and `close()` are plain functions so a callback can call them,
    and must run on the event loop's thread. From another thread — an audio
    library's capture thread, say — schedule them with
    `loop.call_soon_threadsafe(session.write, chunk)`.

    Everything `transcribe_live()` says about when it pays off, the 120 s
    audio cap, the server-side silence abort and errors surfacing mid-upload
    applies here unchanged.

    Example:
        ```python
        async with aai.AsyncSyncTranscriber() as transcriber:
            async with transcriber.open_live(config) as session:
                async for frame in websocket:      # audio frames from a browser
                    session.write(frame)
            print((await session.result()).text)
        ```
    """

    def __init__(
        self,
        start: Callable[
            [AsyncIterator[bytes]], asyncio.Task[types.SyncTranscriptResponse]
        ],
    ) -> None:
        """
        Args:
            start: begins the upload from the given producer and returns the
                task that will hold its outcome. Supplied by the transcriber.
        """
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._closed = False
        self._aborted = False
        self._task = start(self._chunks())

    async def _chunks(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                if self._aborted:
                    raise _Aborted()
                return
            yield chunk

    @property
    def closed(self) -> bool:
        """Whether the audio has ended, by `close()`, `result()` or `abort()`."""

        return self._closed

    def write(self, chunk: bytes) -> None:
        """
        Queues a piece of audio for upload. Never blocks.

        Must be called on the event loop's thread; see the class docstring for
        calling from elsewhere.

        Args:
            chunk: `bytes`, `bytearray` or `memoryview` of audio, in order.

        Raises:
            TypeError: if `chunk` is not bytes-like.
            RuntimeError: if the session is closed.
        """
        if self._closed:
            raise RuntimeError(
                "the live session is closed; no more audio can be written"
            )

        self._queue.put_nowait(_as_bytes(chunk))

    def close(self) -> None:
        """
        Ends the audio. The server transcribes what was sent; `result()`
        returns it. Idempotent, and never blocks.
        """
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    async def result(self) -> types.SyncTranscriptResponse:
        """
        Waits for the transcript, ending the audio first if it is still open.

        Raises:
            SyncTranscriptError: if the request failed, including a rejection
                the server sent while the upload was still in flight.
            RuntimeError: if the session was aborted.
        """
        if self._aborted:
            raise RuntimeError("the live session was aborted; there is no result")

        self.close()

        return await self._task

    async def abort(self) -> None:
        """
        Drops the request without a transcript.

        The connection is closed and nothing the server may still return is
        kept; `result()` raises afterwards. Idempotent, and a no-op once the
        request has completed. Waits until the task has let go of the
        connection. After `close()` the upload is already complete, so
        aborting then waits out the in-flight response instead.
        """
        if self._aborted or self._task.done():
            return

        self._aborted = True
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

        try:
            await self._task
        except Exception:
            pass

    async def __aenter__(self) -> "AsyncLiveSession":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Ends the audio on a clean exit; aborts if the block raised."""

        if exc_type is not None:
            await self.abort()
        else:
            self.close()


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
        api_key: Optional[str] = None,
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
            api_key: The API key to authenticate with. The transcriber builds
                its own `AsyncClient` from it and closes that client on
                `aclose()`. Given alongside `client`, it takes precedence: the
                transcriber builds and owns a client made from a copy of that
                client's settings with the key replaced, and the given client is
                left untouched and stays the caller's to close.

        Raises:
            TypeError: if `config` is not a `SyncTranscriptionConfig`.
        """
        check_config(type(self).__name__, config)

        self._owns_client = client is None or api_key is not None
        self._client = _async_client._resolve_client(client, api_key)
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

        Raises:
            TypeError: if `config` is not a `SyncTranscriptionConfig`.
            SyncTranscriptError: if the request fails.
        """
        check_config(type(self).__name__, config)

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
            # The live timeout: this rides the same streamed connection.
            timeout=self._client.settings.sync_live_http_timeout,
        )

    async def transcribe_live(
        self,
        data: AsyncAudioChunks,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> types.SyncTranscriptResponse:
        """
        Transcribes audio uploaded as it is produced.

        The asyncio counterpart of `SyncTranscriber.transcribe_live`. Where
        `transcribe()` needs the whole clip before it can send anything, this
        starts the request immediately and uploads chunks as they arrive, so
        authorization, the upload and every speech segment but the last resolve
        while the caller is still recording.

        That only pays off when the audio is genuinely still being produced —
        a live microphone, an in-progress call. Streaming a file already on
        disk is slower than `transcribe()`. The saving also needs enough audio
        to have segments to release early: below roughly a minute, only the
        elided upload counts.

        The caller must keep producing: an upload that goes silent for long
        enough is aborted server-side. Stop by ending the iterator, not by
        pausing it.

        Args:
            data: An async iterable of audio chunks, a file object, or a plain
                iterable. A file object is read in a worker thread, so a
                blocking `read()` is fine; a plain iterable is consumed inline
                and so must not block. Raw PCM also requires `sample_rate` and
                `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises:
            TypeError: if `config` is not a `SyncTranscriptionConfig`; if
                `data` is a path or a bytes buffer rather than a stream; or if
                a chunk is not bytes (a file opened in text mode, say).
            SyncTranscriptError: if the request fails. Auth, rate-limit and
                capacity failures can surface part-way through the upload.
            Exception: anything the producer raises mid-upload propagates
                unchanged. The connection is dropped and no transcript is
                returned.

        Example:
            ```python
            async def mic_chunks():
                while recording:
                    yield await stream.read(4096)

            async with aai.AsyncSyncTranscriber() as transcriber:
                result = await transcriber.transcribe_live(mic_chunks())
            ```
        """
        check_config(type(self).__name__, config)

        config = config or self.config
        check_chunks(data, allow_async=True)
        filename, content_type = stream_filename(data, config)

        return await async_api.transcribe_live(
            self._client.http_client,
            base_url=self._client.settings.sync_base_url,
            chunks=data,
            filename=filename,
            audio_content_type=content_type,
            model=config.model,
            config=_config_to_json(config),
            timeout=self._client.settings.sync_live_http_timeout,
        )

    def open_live(
        self,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> AsyncLiveSession:
        """
        Opens a live upload that audio is pushed into.

        The push-style counterpart of `transcribe_live()`, for sources that
        deliver audio through a callback rather than an async iterator. The
        request starts as a task on the running event loop immediately; call
        `session.write(chunk)` as audio arrives, then `await session.result()`
        for the transcript once the speaker stops. See `AsyncLiveSession`.

        Not a coroutine, so it can be used directly as `async with
        transcriber.open_live(config) as session:`. Must be called while the
        event loop is running.

        Args:
            config: Options for this call. If `None`, the transcriber's default
                configuration is used. Raw PCM requires `sample_rate` and
                `channels`.

        Raises:
            TypeError: if `config` is not a `SyncTranscriptionConfig`.
            RuntimeError: if no event loop is running.
        """
        check_config(type(self).__name__, config)
        loop = asyncio.get_running_loop()

        return AsyncLiveSession(
            lambda chunks: loop.create_task(self.transcribe_live(chunks, config=config))
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
