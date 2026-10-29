"""The asyncio counterpart of `client.py`."""

from __future__ import annotations

import asyncio
import functools
from types import TracebackType
from typing import Any, AsyncIterator, Callable, Optional, Type, TypeVar

import httpx
from typing_extensions import Self

from ... import async_client as _async_client
from ..._multipart import _Aborted, _as_bytes
from . import api, async_api
from ._base import AsyncAudioSource, _config_to_json, check_config, resolve_source
from .models import DictationConfig, DictationResponse

_T = TypeVar("_T")


async def _run_in_thread(func: Callable[..., _T], *args: Any) -> _T:
    """Runs a blocking call on the default executor."""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None, func, *args)


class AsyncDictationLiveSession:
    """
    A live dictation that audio is pushed into, for asyncio code.

    Returned by `AsyncDictationTranscriber.open_live()`. The request runs as a
    task on the current event loop from the moment the session opens;
    `write()` hands it audio, `close()` ends the audio, and `await result()`
    waits for the transcript. Built for callback-driven sources — a WebRTC
    track, a websocket handler receiving frames, a telephony media stream —
    where the audio arrives in a callback rather than from an async iterator
    you can hand to `transcribe_live()`.

    `write()` and `close()` are plain functions so a callback can call them,
    and must run on the event loop's thread. From another thread — an audio
    library's capture thread, say — schedule them with
    `loop.call_soon_threadsafe(session.write, chunk)`.

    Everything `transcribe_live()` says about the 120 s audio cap and the
    buffered fallback past it, the server-side silence abort and errors
    surfacing mid-upload applies here unchanged. After a fallback, `result()`
    waits for the whole recording to upload and be transcribed.

    Example:
        ```python
        async with aai.AsyncDictationTranscriber() as transcriber:
            async with transcriber.open_live(config) as session:
                async for frame in websocket:      # audio frames from a browser
                    session.write(frame)
            print((await session.result()).final_text)
        ```
    """

    def __init__(
        self,
        start: Callable[
            [AsyncIterator[bytes], Callable[[], bool]],
            asyncio.Task[DictationResponse],
        ],
    ) -> None:
        """
        Args:
            start: begins the upload from the given producer and returns the
                task that will hold its outcome. Its second argument reports
                whether the session has been aborted, so a buffered fallback is
                never sent for an aborted session. Supplied by the transcriber.
        """
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._closed = False
        self._aborted = False
        self._task = start(self._chunks(), lambda: self._aborted)

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

    async def result(self) -> DictationResponse:
        """
        Waits for the transcript, ending the audio first if it is still open.

        Raises:
            DictationError: if the request failed, including a rejection the
                server sent while the upload was still in flight.
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
        kept; `result()` raises afterwards. No buffered fallback request is
        sent after an abort, and the copy of the audio kept for one is
        dropped. Idempotent, and a no-op once the request has completed.
        Waits until the task has let go of the connection. After `close()` the
        upload is already complete, so aborting then waits out the in-flight
        response instead.
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

    async def __aenter__(self) -> "AsyncDictationLiveSession":
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


class AsyncDictationTranscriber:
    """
    The asyncio counterpart of `DictationTranscriber`: audio in, transcript
    out, one request — without blocking the event loop.

    Same audio sources (an async or plain iterable of chunks, a file object,
    bytes or a local path — no URLs), same `DictationConfig`, same
    `DictationResponse` and `DictationError`, with `transcribe_live()` and
    `warm()` as coroutines and `open_live()` returning an
    `AsyncDictationLiveSession`. Use it in asyncio code (FastAPI, aiohttp,
    voice agents), where the threaded `DictationTranscriber` would block the
    loop. Past the live endpoint's 120 s cap it falls back to the buffered
    endpoint exactly as `DictationTranscriber` does; see
    `fallback_to_buffered` on `__init__`.

    The transcriber owns an HTTP connection pool. Close it with `aclose()`,
    or use the transcriber as an async context manager.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncDictationTranscriber() as transcriber:
                async with transcriber.open_live() as session:
                    async for frame in websocket:
                        session.write(frame)
                print((await session.result()).final_text)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        config: Optional[DictationConfig] = None,
        api_key: Optional[str] = None,
        fallback_to_buffered: bool = True,
    ) -> None:
        """
        Creates an `AsyncDictationTranscriber`.

        Args:
            client: The `AsyncClient` to use. If `None`, the transcriber
                creates one from the global `settings` and closes it on
                `aclose()`. Pass a client to share one pool between
                transcribers.
            config: Default dictation options. Per-call `config` overrides it.
            api_key: The API key to authenticate with. The transcriber builds
                its own `AsyncClient` from it and closes that client on
                `aclose()`. Given alongside `client`, it takes precedence: the
                transcriber builds and owns a client made from a copy of that
                client's settings with the key replaced, and the given client is
                left untouched and stays the caller's to close.
            fallback_to_buffered: Whether a dictation past the live endpoint's
                120 s cap falls back to the buffered endpoint. Applies to
                `transcribe_live()` and `open_live()`. When True (the
                default), the transcriber keeps an in-memory copy of all the
                audio it uploads. If the live request is refused with 413
                `audio_too_large`, it keeps reading the audio source until it
                ends — the producer's iterator is exhausted, or the session is
                closed — then posts the whole recording once to
                `/v1/transcribe` (up to 20 minutes / 256 MB) and returns that
                result like a live one. After a fallback the result waits for
                the whole recording to upload and be transcribed, so a long
                dictation returns more slowly than a live one. The copy costs
                about 32 KB per second of 16 kHz mono audio (about 38 MB at 20
                minutes); past 256 MB the copy is dropped and a 413
                `DictationError` is raised. False keeps no copy and raises the
                live 413 as before.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.
        """
        check_config(type(self).__name__, config)

        self._owns_client = client is None or api_key is not None
        self._client = _async_client._resolve_client(client, api_key)
        self.config = config or DictationConfig()
        self._fallback_to_buffered = fallback_to_buffered

    @property
    def fallback_to_buffered(self) -> bool:
        """Whether audio past the live 120 s cap falls back to `/v1/transcribe`."""

        return self._fallback_to_buffered

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this transcriber sends requests with."""

        return self._client

    async def transcribe_live(
        self,
        data: AsyncAudioSource,
        config: Optional[DictationConfig] = None,
    ) -> DictationResponse:
        """
        Transcribes audio uploaded as it is produced.

        The asyncio counterpart of `DictationTranscriber.transcribe_live`.
        Starts the request immediately and uploads chunks as they arrive, so
        authorization, the upload and every speech segment but the last
        resolve while the caller is still recording. Audio that is already
        complete — bytes, or a local path, which is read off the event loop —
        is accepted as well and sent as a single chunk over the same
        connection.

        The caller must keep producing: an upload that goes silent for long
        enough is aborted server-side. Stop by ending the iterator, not by
        pausing it. The live endpoint caps a request at 120 s of audio. Past
        it, unless the transcriber was built with `fallback_to_buffered=False`,
        the iterator is read to its end and the whole recording is posted to
        the buffered endpoint (up to 20 minutes), whose result is returned.
        That result waits for the whole recording to upload and be
        transcribed, so a long dictation returns more slowly than a live one.

        Args:
            data: An async iterable of audio chunks, a plain iterable, a file
                object, raw audio bytes, or a local file path. A file object is
                read in a worker thread, so a blocking `read()` is fine; a
                plain iterable is consumed inline and so must not block. Raw
                PCM also requires `sample_rate` and `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises:
            TypeError: if `config` is not a `DictationConfig`; if `data` is of
                an unsupported type; or if a chunk is not bytes (a file opened
                in text mode, say).
            ValueError: for a URL, or raw PCM without both `sample_rate` and
                `channels`.
            DictationError: if the request fails. Auth, rate-limit, size and
                capacity failures can surface part-way through the upload.
                After a fallback, it is the buffered request's own error (a
                413 past 20 minutes, say), or a 413 `audio_too_large` once the
                recording passes 256 MB.
            Exception: anything the producer raises mid-upload propagates
                unchanged. The connection is dropped and no transcript is
                returned.

        Example:
            ```python
            async def mic_chunks():
                while recording:
                    yield await stream.read(4096)

            async with aai.AsyncDictationTranscriber() as transcriber:
                result = await transcriber.transcribe_live(mic_chunks())
            ```
        """
        check_config(type(self).__name__, config)

        return await self._transcribe_live(data, config)

    async def _transcribe_live(
        self,
        data: AsyncAudioSource,
        config: Optional[DictationConfig],
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> DictationResponse:
        config = config or self.config
        # Resolving may read a whole file for a path source; keep that off
        # the loop.
        chunks, filename, content_type = await _run_in_thread(
            functools.partial(
                resolve_source, data, config, type(self).__name__, allow_async=True
            )
        )

        return await async_api.transcribe_live(
            self._client.http_client,
            base_url=self._client.settings.dictation_base_url,
            chunks=chunks,
            filename=filename,
            audio_content_type=content_type,
            config=_config_to_json(config),
            timeout=self._client.settings.dictation_http_timeout,
            fallback_to_buffered=self._fallback_to_buffered,
            cancelled=cancelled,
        )

    def open_live(
        self,
        config: Optional[DictationConfig] = None,
    ) -> AsyncDictationLiveSession:
        """
        Opens a live dictation that audio is pushed into.

        The push-style counterpart of `transcribe_live()`, for sources that
        deliver audio through a callback rather than an async iterator. The
        request starts as a task on the running event loop immediately; call
        `session.write(chunk)` as audio arrives, then `await session.result()`
        for the transcript once the speaker stops. See
        `AsyncDictationLiveSession`.

        Past the live endpoint's 120 s cap the session falls back the way
        `transcribe_live()` does: keep writing until the speaker stops, and
        `result()` returns the buffered endpoint's transcript of the whole
        recording, more slowly than a live result.

        Not a coroutine, so it can be used directly as `async with
        transcriber.open_live(config) as session:`. Must be called while the
        event loop is running.

        Args:
            config: Options for this call. If `None`, the transcriber's default
                configuration is used. Raw PCM requires `sample_rate` and
                `channels`.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.
            RuntimeError: if no event loop is running.
        """
        check_config(type(self).__name__, config)
        loop = asyncio.get_running_loop()

        return AsyncDictationLiveSession(
            lambda chunks, cancelled: loop.create_task(
                self._transcribe_live(chunks, config, cancelled)
            )
        )

    async def warm(self) -> bool:
        """
        Opens the connection to the Dictation API ahead of time.

        A request that opens its connection on demand pays the full DNS + TCP
        + TLS handshake before the first audio byte can leave — one network
        round trip that, for a distant client, is a noticeable share of a
        short dictation. Awaiting `warm()` as soon as you know audio is coming
        — e.g. via `asyncio.create_task(transcriber.warm())` when the user
        reaches for the record button — spends that setup early: the next
        request reuses the already-open connection.

        The warmed connection is reused while it stays in the HTTP pool —
        `settings.keepalive_expiry` seconds (httpx's 5s default unless
        raised). Call `warm()` shortly before the request, or raise
        `keepalive_expiry` so a single call covers a longer pause. `warm()` is
        idempotent and cheap, so calling it again to refresh the connection is
        fine.

        Returns:
            True once the connection is open (any HTTP response — even a
            non-200 — means the socket is established); False if the
            connection could not be opened (transport error).
        """
        settings = self._client.settings
        url = settings.dictation_base_url.rstrip("/") + api.ENDPOINT_WARM
        try:
            await self._client.http_client.get(
                url,
                timeout=min(settings.dictation_http_timeout, 10.0),
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
