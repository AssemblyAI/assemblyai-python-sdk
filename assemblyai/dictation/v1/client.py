from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Optional

import httpx

from ... import client as _client
from . import api
from ._base import AudioInput, _DictationTranscriberImpl, check_config
from .models import DictationConfig, DictationResponse


class DictationTranscriber:
    """
    Transcribes dictated audio in one request: audio in, transcript out.

    Targets the Dictation API (`dictation.assemblyai.com`), tuned for short
    dictated clips. Like `SyncTranscriber` there is no job id and no polling,
    and it accepts a local file path, raw bytes, or a binary file object —
    but not a URL. Beyond the transcript it can run a follow-up LLM pass:
    set `llm_instruction` on the config and read `result.final_text`.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        result = aai.DictationTranscriber().transcribe("./note.wav")
        print(result.text)
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
            max_workers: Thread pool size for `transcribe_async`. Defaults to
                the CPU count minus one.
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

    def transcribe(
        self,
        data: AudioInput,
        config: Optional[DictationConfig] = None,
    ) -> DictationResponse:
        """
        Transcribes audio and returns the finished transcript.

        Args:
            data: A local file path, raw audio bytes, or a binary file object.
                Raw PCM also requires `sample_rate` and `channels` on the config.
            config: Options for this call. If `None`, the transcriber's default
                configuration is used.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.
            DictationError: if the request fails.
        """
        check_config(type(self).__name__, config)

        return self._impl.transcribe(data=data, config=config)

    def transcribe_async(
        self,
        data: AudioInput,
        config: Optional[DictationConfig] = None,
    ) -> "concurrent.futures.Future[DictationResponse]":
        """
        Transcribes audio on a worker thread.

        Returns a `concurrent.futures.Future` (not an asyncio coroutine); call
        `.result()` to block for the transcript. Useful for fanning out a
        handful of clips concurrently.

        Raises:
            TypeError: if `config` is not a `DictationConfig`.
        """
        check_config(type(self).__name__, config)

        return self._executor.submit(
            self._impl.transcribe,
            data=data,
            config=config,
        )

    def warm(self) -> bool:
        """
        Opens the connection to the Dictation API ahead of time.

        The Dictation API is a single request/response, so a `transcribe()`
        that opens its connection on demand pays the full DNS + TCP + TLS
        handshake on the critical path — one network round trip that, for a
        distant client, can rival the transcription itself. Calling `warm()`
        as soon as you know audio is coming — typically while the clip is
        still being recorded — spends that setup concurrently: the next
        `transcribe()` reuses the already-open connection.

        The warmed connection is reused while it stays in the HTTP pool —
        `settings.keepalive_expiry` seconds (httpx's 5s default unless
        raised). Call `warm()` shortly before `transcribe()`, or raise
        `keepalive_expiry` so a single call covers a whole in-progress
        recording. `warm()` is idempotent and cheap, so calling it again to
        refresh the connection is fine.

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
        """Shuts down the worker-thread pool used by `transcribe_async`."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "DictationTranscriber":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
