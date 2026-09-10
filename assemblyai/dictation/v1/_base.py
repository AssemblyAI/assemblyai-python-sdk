from __future__ import annotations

import os
from typing import AsyncIterable, BinaryIO, Iterable, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

from ... import _audio
from ... import client as _client
from ..._multipart import AsyncAudioChunks, AudioChunks
from . import api
from .models import DictationConfig, DictationResponse

__all__ = [
    "AsyncAudioSource",
    "AudioSource",
    "_DictationTranscriberImpl",
    "_config_to_json",
    "check_config",
    "resolve_source",
]

# Audio for a dictation request. The live endpoint is the only one this
# client opens, so it takes every shape audio comes in: a producer that is
# still recording (an iterable of chunks, or a file object read as it fills)
# and audio that is already complete (bytes, or a local path), which goes out
# over the same connection as a single chunk.
AudioSource = Union[
    str, bytes, bytearray, "os.PathLike[str]", BinaryIO, Iterable[bytes]
]
AsyncAudioSource = Union[AudioSource, AsyncIterable[bytes]]

# The Dictation API decodes raw PCM and the common container formats; the
# extension picks the Content-Type that tells the server which decoder to use.
# The live route accepts WAV and PCM today: a compressed format is posted with
# its true type so the server's rejection names it, rather than mislabelled as
# WAV and failing to decode.
_CONTENT_TYPES: Mapping[str, str] = {
    ".wav": "audio/wav",
    ".pcm": "audio/pcm",
    ".raw": "audio/pcm",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".mp4": "audio/mp4",
    ".m4a": "audio/x-m4a",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def check_config(owner: str, config: Optional[DictationConfig]) -> None:
    """
    Raises unless `config` is a `DictationConfig` or `None`.

    `SyncTranscriptionConfig` and the job API's `TranscriptionConfig` are
    different, non-interchangeable types, so passing one here is a mistake
    worth naming.

    Args:
        owner: the class to name in the message, e.g. `DictationTranscriber`.
        config: the configuration to check.
    """

    _audio.check_config(owner, config, DictationConfig)


def _config_to_json(config: DictationConfig) -> Optional[dict]:
    """Serializes the config to the JSON `config` part."""

    return _audio._config_to_json(config)


def resolve_source(
    data: object,
    config: DictationConfig,
    owner: str,
    *,
    allow_async: bool = False,
) -> Tuple[Union[AudioChunks, AsyncAudioChunks], str, str]:
    """
    Turns any accepted audio source into chunks for the live upload, and
    decides the audio part's filename and Content-Type.

    Complete audio — `bytes`, or a local path, which is read here — becomes a
    single chunk. A file object or an iterable is handed on untouched and read
    as the upload proceeds. PCM is selected when the source has a `.pcm`/`.raw`
    extension or when `sample_rate`/`channels` are set on the config (the
    fields the Dictation API requires only for raw PCM) — and both must then be
    present. Any other extension maps to its container Content-Type, and audio
    with no usable extension is posted as WAV. URLs are rejected — the
    Dictation API has no URL ingestion.

    Args:
        data: the audio source.
        config: the configuration for this call.
        owner: the class to name in error messages, so each client reports
            its own name.
        allow_async: whether an object exposing only `__aiter__` is acceptable,
            i.e. whether the caller is `AsyncDictationTranscriber`.

    Returns: `(chunks, filename, content_type)`.

    Raises:
        ValueError: for a URL, or raw PCM missing `sample_rate` or `channels`.
        TypeError: for a source of an unsupported type, or an async iterable
            handed to the threaded transcriber.
    """
    suffix = ""
    filename: Optional[str] = None
    chunks: Union[AudioChunks, AsyncAudioChunks]

    if isinstance(data, (bytes, bytearray)):
        chunks = (bytes(data),)
    elif isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if urlparse(path).scheme in ("http", "https"):
            raise ValueError(
                f"{owner} does not accept URLs. Pass a local file path, audio "
                "bytes, a binary file object or an iterable of audio chunks, or "
                "use aai.Transcriber for URL transcription."
            )
        with open(path, "rb") as f:
            chunks = (f.read(),)
        filename = os.path.basename(path)
        suffix = os.path.splitext(path)[1].lower()
    elif hasattr(data, "read"):
        chunks = data  # type: ignore[assignment]
        name = getattr(data, "name", None)
        if isinstance(name, str) and name:
            filename = os.path.basename(name)
            suffix = os.path.splitext(name)[1].lower()
    elif hasattr(data, "__iter__"):
        chunks = data  # type: ignore[assignment]
    elif hasattr(data, "__aiter__"):
        if not allow_async:
            raise TypeError(
                "DictationTranscriber.transcribe_live() cannot consume an async "
                "iterable. Use AsyncDictationTranscriber.transcribe_live(), or "
                "hand it a plain iterable or file object."
            )
        chunks = data  # type: ignore[assignment]
    else:
        raise TypeError(f"unsupported audio source type: {type(data).__name__}")

    resolved_filename, content_type = _audio.resolve_format(
        suffix=suffix,
        filename=filename,
        sample_rate=config.sample_rate,
        channels=config.channels,
        config_name="DictationConfig",
        content_types=_CONTENT_TYPES,
    )

    return chunks, resolved_filename, content_type


class _DictationTranscriberImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
        config: DictationConfig,
        owner: str,
    ) -> None:
        self._client = client
        self.config = config
        self._owner = owner

    def transcribe_live(
        self,
        *,
        data: AudioSource,
        config: Optional[DictationConfig],
    ) -> DictationResponse:
        config = config or self.config
        chunks, filename, content_type = resolve_source(data, config, self._owner)

        return api.transcribe_live(
            self._client.http_client,
            base_url=self._client.settings.dictation_base_url,
            chunks=chunks,  # type: ignore[arg-type]
            filename=filename,
            audio_content_type=content_type,
            config=_config_to_json(config),
            timeout=self._client.settings.dictation_http_timeout,
        )
