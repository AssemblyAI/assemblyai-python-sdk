from __future__ import annotations

import concurrent.futures
import os
from typing import BinaryIO, Optional, Tuple, Union
from urllib.parse import urlparse

from . import client as _client
from . import sync_api, types

AudioInput = Union[str, bytes, bytearray, "os.PathLike[str]", BinaryIO]

# Extensions that signal raw S16LE PCM rather than a WAV container.
_PCM_SUFFIXES = (".pcm", ".raw")


def _resolve_audio(
    data: AudioInput,
    config: types.SyncTranscriptionConfig,
) -> Tuple[bytes, str, str]:
    """
    Reads the audio input into bytes and decides its multipart Content-Type.

    PCM is selected when the source has a `.pcm`/`.raw` extension or when
    `sample_rate`/`channels` are set on the config (the fields the sync API
    requires only for raw PCM) — and both must then be present. Everything
    else is treated as a WAV container. URLs are rejected — the sync API has
    no URL ingestion.

    Returns: `(audio_bytes, filename, content_type)`.
    """
    suffix = ""
    filename: Optional[str] = None

    if isinstance(data, (bytes, bytearray)):
        audio = bytes(data)
    elif isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if urlparse(path).scheme in ("http", "https"):
            raise ValueError(
                "SyncTranscriber does not accept URLs. Pass a local file path or "
                "audio bytes, or use aai.Transcriber for URL/async transcription."
            )
        with open(path, "rb") as f:
            audio = f.read()
        filename = os.path.basename(path)
        suffix = os.path.splitext(path)[1].lower()
    elif hasattr(data, "read"):
        audio = data.read()
        name = getattr(data, "name", None)
        if name:
            filename = os.path.basename(name)
            suffix = os.path.splitext(name)[1].lower()
    else:
        raise TypeError(f"unsupported audio input type: {type(data).__name__}")

    wants_pcm = config.sample_rate is not None or config.channels is not None
    is_pcm = suffix in _PCM_SUFFIXES or wants_pcm
    if is_pcm and (config.sample_rate is None or config.channels is None):
        raise ValueError(
            "raw PCM audio requires both sample_rate and channels in "
            "SyncTranscriptionConfig"
        )

    content_type = "audio/pcm" if is_pcm else "audio/wav"
    if not filename:
        filename = "audio.pcm" if is_pcm else "audio.wav"

    return audio, filename, content_type


def _config_to_json(config: types.SyncTranscriptionConfig) -> Optional[dict]:
    """Serializes the config to the JSON `config` part, dropping the routing model."""
    data = config.dict(exclude_none=True)
    data.pop("model", None)
    return data or None


class _SyncTranscriberImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
        config: types.SyncTranscriptionConfig,
    ) -> None:
        self._client = client
        self.config = config

    def transcribe(
        self,
        *,
        data: AudioInput,
        config: Optional[types.SyncTranscriptionConfig],
    ) -> types.SyncTranscriptResponse:
        config = config or self.config
        audio, filename, content_type = _resolve_audio(data, config)
        return sync_api.transcribe(
            self._client.http_client,
            base_url=self._client.settings.sync_base_url,
            audio=audio,
            filename=filename,
            audio_content_type=content_type,
            model=config.model,
            config=_config_to_json(config),
            timeout=self._client.settings.sync_http_timeout,
        )


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
