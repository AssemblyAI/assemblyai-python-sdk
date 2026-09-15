from __future__ import annotations

import concurrent.futures
import os
import queue
from typing import Any, Callable, Iterator, Optional

import httpx

from ... import client as _client
from ..._multipart import _Aborted, _as_bytes
from . import api
from ._base import AudioSource, _DictationTranscriberImpl, check_config
from .models import DictationConfig, DictationResponse


class DictationLiveSession:
    """
    A live dictation that audio is pushed into.

    Returned by `DictationTranscriber.open_live()`. The request starts on one
    of the transcriber's worker threads the moment the session opens;
    `write()` hands it audio, `close()` ends the audio, and `result()` waits
    for the transcript. Built for callback-driven sources — a microphone
    library, a WebRTC track, a telephony media stream — where the audio
    arrives in a callback rather than from an iterator you can hand to
    `transcribe_live()`.

    Everything `transcribe_live()` says about the 120 s audio cap, the
    server-side silence abort and errors surfacing mid-upload applies here
    unchanged.

    Example:
        ```python
        import sounddevice as sd

        config = aai.DictationConfig(sample_rate=16000, channels=1)

        with aai.DictationTranscriber() as transcriber:
            with transcriber.open_live(config) as session:
                stream = sd.RawInputStream(
                    samplerate=16000, channels=1, dtype="int16",
                    callback=lambda data, *_: session.write(bytes(data)),
                )
                with stream:
                    input("Dictating, press Enter to stop... ")
            print(session.result().final_text)
        ```
    """

    def __init__(
        self,
        start: Callable[
            [Iterator[bytes]], concurrent.futures.Future[DictationResponse]
        ],
    ) -> None:
        """
        Args:
            start: begins the upload from the given producer and returns the
                future that will hold its outcome. Supplied by the transcriber.
        """
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False
        self._aborted = False
        self._future = start(self._chunks())

    def _chunks(self) -> Iterator[bytes]:
        while True:
            chunk = self._queue.get()
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
        Queues a piece of audio for upload.

        Safe to call from any thread, including an audio library's capture
        callback; it never blocks.

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

        self._queue.put(_as_bytes(chunk))

    def close(self) -> None:
        """
        Ends the audio. The server transcribes what was sent; `result()`
        returns it. Idempotent, and never blocks.
        """
        if not self._closed:
            self._closed = True
            self._queue.put(None)

    def result(self) -> DictationResponse:
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

        return self._future.result()

    def abort(self) -> None:
        """
        Drops the request without a transcript.

        The connection is closed and nothing the server may still return is
        kept; `result()` raises afterwards. Idempotent, and a no-op once the
        request has completed. Blocks until the worker thread has let go of
        the connection. After `close()` the upload is already complete, so
        aborting then waits out the in-flight response instead.
        """
        if self._aborted or self._future.done():
            return

        self._aborted = True
        if not self._closed:
            self._closed = True
            self._queue.put(None)

        try:
            self._future.result()
        except Exception:
            pass

    def __enter__(self) -> "DictationLiveSession":
        return self

    def __exit__(self, exc_type: Any, *_exc: Any) -> None:
        """Ends the audio on a clean exit; aborts if the block raised."""

        if exc_type is not None:
            self.abort()
        else:
            self.close()


class DictationTranscriber:
    """
    Transcribes dictated audio as it is spoken: audio in, transcript out, one
    request.

    Targets the Dictation API (`dictation.assemblyai.com`), tuned for short
    dictated clips. The audio is uploaded while it is still being produced —
    from a callback with `open_live()`, or from an iterator or file object
    with `transcribe_live()` — so the upload and every speech segment but the
    last are done by the time the speaker stops. Audio you already hold whole
    (bytes, or a local path) goes to `transcribe_live()` too; it travels the
    same connection as a single chunk. Like `SyncTranscriber` there is no job
    id and no polling, and no URL ingestion. Beyond the transcript it can run
    a follow-up LLM pass: set `llm_instruction` on the config and read
    `result.final_text`.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        with aai.DictationTranscriber() as transcriber:
            with transcriber.open_live() as session:
                start_capture(on_audio=session.write)
                wait_until_done()
            print(session.result().final_text)
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[DictationConfig] = None,
        max_workers: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates a `DictationTranscriber`.

        Args:
            client: The HTTP client to use. Defaults to the shared default client.
            config: Default dictation options. Per-call `config` overrides it.
            max_workers: Thread pool size for `open_live()` sessions, each of
                which runs its request on a worker thread. Defaults to the CPU
                count minus one.
            api_key: The API key to authenticate with. Builds a `Client` for this
                transcriber. Given alongside `client`, it takes precedence: the
                transcriber builds its own client from a copy of that client's
                settings with the key replaced, and the given client is left
                untouched.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.
        """
        check_config(type(self).__name__, config)

        self._client = _client._resolve_client(client, api_key)
        self._impl = _DictationTranscriberImpl(
            client=self._client,
            config=config or DictationConfig(),
            owner=type(self).__name__,
        )

        if not max_workers:
            cpu_count = os.cpu_count()
            max_workers = max(1, cpu_count - 1) if cpu_count else 1

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )

    @property
    def config(self) -> DictationConfig:
        """The default configuration of the `DictationTranscriber`."""
        return self._impl.config

    @config.setter
    def config(self, config: DictationConfig) -> None:
        check_config(type(self).__name__, config)

        self._impl.config = config

    def transcribe_live(
        self,
        data: AudioSource,
        config: Optional[DictationConfig] = None,
    ) -> DictationResponse:
        """
        Transcribes audio uploaded as it is produced.

        Starts the request immediately and uploads chunks as they arrive, so
        authorization, the upload and every speech segment but the last
        resolve while the caller is still recording. What is left to wait for
        once they stop is the final segment, and the LLM pass if one was asked
        for. Audio that is already complete — bytes, or a local path — is
        accepted as well and sent as a single chunk over the same connection.

        The caller must keep producing: an upload that goes silent for long
        enough is aborted server-side. Stop by ending the iterator, not by
        pausing it. The service caps a request at 120 s of audio.

        Args:
            data: An iterable of audio chunks, a binary file object read as it
                fills, raw audio bytes, or a local file path. Raw PCM also
                requires `sample_rate` and `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises:
            TypeError: if `config` is not a `DictationConfig`; if `data` is an
                async iterable or an unsupported type; or if a chunk is not
                bytes (a file opened in text mode, say).
            ValueError: for a URL, or raw PCM without both `sample_rate` and
                `channels`.
            DictationError: if the request fails. Auth, rate-limit, size and
                capacity failures can surface part-way through the upload.
            Exception: anything the producer raises mid-upload propagates
                unchanged. The connection is dropped and no transcript is
                returned.

        Example:
            ```python
            def mic_chunks():
                while recording:
                    yield stream.read(4096)

            result = aai.DictationTranscriber().transcribe_live(mic_chunks())
            ```
        """
        check_config(type(self).__name__, config)

        return self._impl.transcribe_live(data=data, config=config)

    def open_live(
        self,
        config: Optional[DictationConfig] = None,
    ) -> DictationLiveSession:
        """
        Opens a live dictation that audio is pushed into.

        The push-style counterpart of `transcribe_live()`, for sources that
        deliver audio through a callback rather than an iterator. The request
        starts on a worker thread immediately; call `session.write(chunk)` from
        the callback, then `session.result()` for the transcript once the
        speaker stops. See `DictationLiveSession`.

        Args:
            config: Options for this call. If `None`, the transcriber's default
                configuration is used. Raw PCM requires `sample_rate` and
                `channels`.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.

        Example:
            ```python
            with transcriber.open_live(config) as session:
                start_capture(on_audio=session.write)
                wait_until_done()
            result = session.result()
            ```
        """
        check_config(type(self).__name__, config)

        return DictationLiveSession(
            lambda chunks: self._executor.submit(
                self._impl.transcribe_live, data=chunks, config=config
            )
        )

    def warm(self) -> bool:
        """
        Opens the connection to the Dictation API ahead of time.

        A request that opens its connection on demand pays the full DNS + TCP
        + TLS handshake before the first audio byte can leave — one network
        round trip that, for a distant client, is a noticeable share of a
        short dictation. Calling `warm()` as soon as you know audio is coming
        — when the user reaches for the record button, say — spends that setup
        early: the next request reuses the already-open connection.

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
            self._client.http_client.get(
                url,
                timeout=min(settings.dictation_http_timeout, 10.0),
            )
        except httpx.HTTPError:
            return False
        return True

    def close(self) -> None:
        """Shuts down the worker-thread pool that `open_live()` sessions run on."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "DictationTranscriber":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
