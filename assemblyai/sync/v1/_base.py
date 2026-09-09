from __future__ import annotations

import os
from typing import BinaryIO, Optional, Tuple, Union
from urllib.parse import urlparse

from ... import client as _client
from ... import types
from . import api
from ._multipart import AudioChunks

AudioInput = Union[str, bytes, bytearray, "os.PathLike[str]", BinaryIO]

# Extensions that signal raw S16LE PCM rather than a WAV container.
_PCM_SUFFIXES = (".pcm", ".raw")


def check_config(owner: str, config: Optional[types.SyncTranscriptionConfig]) -> None:
    """
    Raises unless `config` is a `SyncTranscriptionConfig` or `None`.

    The job API's `TranscriptionConfig` is a different, non-interchangeable
    type, so passing it here is a mistake worth naming.

    Args:
        owner: the class to name in the message, e.g. `SyncTranscriber`.
        config: the configuration to check.
    """

    if config is not None and not isinstance(config, types.SyncTranscriptionConfig):
        raise TypeError(
            f"{owner} expects SyncTranscriptionConfig, got {type(config).__name__}. "
            "Use aai.SyncTranscriptionConfig."
        )


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

    resolved_filename, content_type = resolve_format(config, suffix, filename)

    return audio, resolved_filename, content_type


def resolve_format(
    config: types.SyncTranscriptionConfig,
    suffix: str = "",
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Decides the multipart filename and Content-Type for the audio part.

    PCM is selected when `suffix` is a PCM extension or when
    `sample_rate`/`channels` are set on the config — the fields the sync API
    requires only for raw PCM — and both must then be present. Everything else
    is treated as a WAV container. Needs no audio bytes, so it serves the
    streamed path as well as the buffered one.

    Args:
        config: the transcription options.
        suffix: the source's lowercased file extension, if any.
        filename: the name for the multipart part; defaulted when absent.

    Returns: `(filename, content_type)`.
    """
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

    return filename, content_type


def _config_to_json(config: types.SyncTranscriptionConfig) -> Optional[dict]:
    """Serializes the config to the JSON `config` part, dropping the routing model."""
    data = config.dict(exclude_none=True)
    data.pop("model", None)
    return data or None


def check_chunks(data: object, *, allow_async: bool = False) -> None:
    """
    Raises unless `data` can be streamed.

    Names the mistakes worth catching early — a whole audio buffer, which
    belongs in `transcribe()`; a path, which the streaming path cannot open on
    the caller's behalf without deciding when to read it; and an async
    iterable handed to the synchronous transcriber, which cannot drive it.

    Args:
        data: the candidate audio source.
        allow_async: whether an object exposing only `__aiter__` is acceptable,
            i.e. whether the caller is `AsyncSyncTranscriber`.
    """
    if isinstance(data, (bytes, bytearray)):
        raise TypeError(
            "transcribe_live() expects an iterable of audio chunks or a file "
            "object, not audio bytes. Audio you already hold whole should go to "
            "transcribe(), which is faster for it."
        )

    if isinstance(data, (str, os.PathLike)):
        raise TypeError(
            "transcribe_live() expects an iterable of audio chunks or a file "
            "object, not a path. Open the file and pass the file object, or use "
            "transcribe() to let the SDK read it."
        )

    if hasattr(data, "read") or hasattr(data, "__iter__"):
        return

    if hasattr(data, "__aiter__"):
        if allow_async:
            return
        raise TypeError(
            "SyncTranscriber.transcribe_live() cannot consume an async "
            "iterable. Use AsyncSyncTranscriber.transcribe_live(), or hand it "
            "a plain iterable or file object."
        )

    raise TypeError(f"unsupported audio stream type: {type(data).__name__}")


def stream_filename(
    data: object, config: types.SyncTranscriptionConfig
) -> Tuple[str, str]:
    """Resolves the audio part's filename and Content-Type for a stream."""

    name = getattr(data, "name", None)
    filename = os.path.basename(name) if isinstance(name, str) and name else None
    suffix = os.path.splitext(filename)[1].lower() if filename else ""

    return resolve_format(config, suffix, filename)


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
        return api.transcribe(
            self._client.http_client,
            base_url=self._client.settings.sync_base_url,
            audio=audio,
            filename=filename,
            audio_content_type=content_type,
            model=config.model,
            config=_config_to_json(config),
            timeout=self._client.settings.sync_http_timeout,
        )

    def transcribe_live(
        self,
        *,
        data: AudioChunks,
        config: Optional[types.SyncTranscriptionConfig],
    ) -> types.SyncTranscriptResponse:
        config = config or self.config
        check_chunks(data)
        filename, content_type = stream_filename(data, config)

        return api.transcribe_live(
            self._client.http_client,
            base_url=self._client.settings.sync_base_url,
            chunks=data,
            filename=filename,
            audio_content_type=content_type,
            model=config.model,
            config=_config_to_json(config),
            timeout=self._client.settings.sync_live_http_timeout,
        )
