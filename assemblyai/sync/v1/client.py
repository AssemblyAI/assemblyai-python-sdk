from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Optional

import httpx

from ... import client as _client
from ... import types
from . import api
from ._base import AudioInput, _SyncTranscriberImpl


class SyncTranscriber:
    """
    Transcribes audio synchronously: audio in, transcript out, one request.

    Unlike `Transcriber` (which submits a job to the async API and polls for
    completion), `SyncTranscriber` posts the audio to the sync API and returns
    the finished `SyncTranscriptResponse` directly. There is no job id or
    status to poll. Accepts a local file path, raw bytes, or a binary file
    object — but not a URL.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        result = aai.SyncTranscriber().transcribe("./call.wav")
        print(result.text)
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[types.SyncTranscriptionConfig] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Creates a `SyncTranscriber`.

        Args:
            client: The HTTP client to use. Defaults to the shared default client.
            config: Default transcription options. Per-call `config` overrides it.
            max_workers: Thread pool size for `transcribe_async`. Defaults to
                the CPU count minus one.
        """
        self._client = client or _client.Client.get_default()
        self._impl = _SyncTranscriberImpl(
            client=self._client,
            config=config or types.SyncTranscriptionConfig(),
        )

        if not max_workers:
            cpu_count = os.cpu_count()
            max_workers = max(1, cpu_count - 1) if cpu_count else 1

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )

    @property
    def config(self) -> types.SyncTranscriptionConfig:
        """The default configuration of the `SyncTranscriber`."""
        return self._impl.config

    @config.setter
    def config(self, config: types.SyncTranscriptionConfig) -> None:
        self._impl.config = config

    def transcribe(
        self,
        data: AudioInput,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> types.SyncTranscriptResponse:
        """
        Transcribes audio and returns the finished transcript.

        Args:
            data: A local file path, raw audio bytes, or a binary file object.
                Raw PCM also requires `sample_rate` and `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises: `SyncTranscriptError` if the request fails.
        """
        return self._impl.transcribe(data=data, config=config)

    def transcribe_async(
        self,
        data: AudioInput,
        config: Optional[types.SyncTranscriptionConfig] = None,
    ) -> "concurrent.futures.Future[types.SyncTranscriptResponse]":
        """
        Transcribes audio on a worker thread.

        Returns a `concurrent.futures.Future` (not an asyncio coroutine); call
        `.result()` to block for the transcript. Useful for fanning out a
        handful of files concurrently.
        """
        return self._executor.submit(
            self._impl.transcribe,
            data=data,
            config=config,
        )

    def warm(self) -> bool:
        """
        Opens the connection to the sync API ahead of time.

        The sync API is a single request/response, so a `transcribe()` that
        opens its connection on demand pays the full DNS + TCP + TLS handshake
        on the critical path — one network round trip that, for a distant
        client, can rival the transcription itself. Calling `warm()` as soon as
        you know audio is coming — typically while the clip is still being
        recorded — spends that setup concurrently: the next `transcribe()`
        reuses the already-open connection.

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
            self._client.http_client.get(
                url,
                headers={api.MODEL_HEADER: self.config.model},
                timeout=min(settings.sync_http_timeout, 10.0),
            )
        except httpx.HTTPError:
            return False
        return True

    def close(self) -> None:
        """Shuts down the worker-thread pool used by `transcribe_async`."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "SyncTranscriber":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
