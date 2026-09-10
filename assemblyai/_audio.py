"""Audio-input helpers shared by the single-request transcription products.

Both the sync API and the Dictation API take audio as a multipart upload
rather than a URL, so they read the same input shapes (path, bytes, binary
file object), decide a multipart `Content-Type` the same way, and serialize
their config to the same JSON part. The product packages wrap these with
their own config type and error wording.
"""

from __future__ import annotations

import os
from typing import Any, BinaryIO, Mapping, Optional, Tuple, Type, Union
from urllib.parse import urlparse

AudioInput = Union[str, bytes, bytearray, "os.PathLike[str]", BinaryIO]

# Extensions that signal raw S16LE PCM rather than a container format.
_PCM_SUFFIXES = (".pcm", ".raw")

# Content-Type used when the extension is unknown or absent.
_DEFAULT_CONTENT_TYPE = "audio/wav"


def check_config(owner: str, config: Optional[Any], expected_type: Type[Any]) -> None:
    """
    Raises unless `config` is an instance of `expected_type` or `None`.

    The job API's `TranscriptionConfig` is a different, non-interchangeable
    type, so passing it here is a mistake worth naming.

    Args:
        owner: the class to name in the message, e.g. `SyncTranscriber`.
        config: the configuration to check.
        expected_type: the configuration class `owner` accepts.
    """

    if config is not None and not isinstance(config, expected_type):
        raise TypeError(
            f"{owner} expects {expected_type.__name__}, got {type(config).__name__}. "
            f"Use aai.{expected_type.__name__}."
        )


def _resolve_audio(
    data: AudioInput,
    *,
    sample_rate: Optional[int],
    channels: Optional[int],
    owner: str,
    config_name: str,
    content_types: Mapping[str, str],
) -> Tuple[bytes, str, str]:
    """
    Reads the audio input into bytes and decides its multipart Content-Type.

    PCM is selected when the source has a `.pcm`/`.raw` extension or when
    `sample_rate`/`channels` are set on the config (the fields these APIs
    require only for raw PCM) — and both must then be present. Any other
    input takes the Content-Type `content_types` maps its extension to, and
    `audio/wav` when the extension is unknown or absent. URLs are rejected —
    neither API has URL ingestion.

    Args:
        data: a local file path, raw bytes, or a binary file object.
        sample_rate: the config's `sample_rate`, or `None`.
        channels: the config's `channels`, or `None`.
        owner: the class to name in error messages, e.g. `SyncTranscriber`.
        config_name: the config class to name in error messages.
        content_types: extension (lowercased, with the dot) → Content-Type.

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
                f"{owner} does not accept URLs. Pass a local file path or "
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

    resolved_filename, content_type = resolve_format(
        suffix=suffix,
        filename=filename,
        sample_rate=sample_rate,
        channels=channels,
        config_name=config_name,
        content_types=content_types,
    )

    return audio, resolved_filename, content_type


def resolve_format(
    *,
    suffix: str,
    filename: Optional[str],
    sample_rate: Optional[int],
    channels: Optional[int],
    config_name: str,
    content_types: Mapping[str, str],
) -> Tuple[str, str]:
    """
    Decides the multipart filename and Content-Type for the audio part.

    PCM is selected when `suffix` is a PCM extension or when `sample_rate` /
    `channels` are set — the fields these APIs require only for raw PCM — and
    both must then be present. Any other suffix takes the Content-Type
    `content_types` maps it to, and `audio/wav` when it is unknown or absent.
    Needs no audio bytes, so it serves a streamed upload as well as a buffered
    one.

    Args:
        suffix: the source's lowercased file extension (with the dot), or "".
        filename: the name for the multipart part; defaulted when absent.
        sample_rate: the config's `sample_rate`, or `None`.
        channels: the config's `channels`, or `None`.
        config_name: the config class to name in error messages.
        content_types: extension (lowercased, with the dot) → Content-Type.

    Returns: `(filename, content_type)`.
    """
    wants_pcm = sample_rate is not None or channels is not None
    is_pcm = suffix in _PCM_SUFFIXES or wants_pcm
    if is_pcm and (sample_rate is None or channels is None):
        raise ValueError(
            f"raw PCM audio requires both sample_rate and channels in {config_name}"
        )

    if is_pcm:
        content_type = "audio/pcm"
    else:
        content_type = content_types.get(suffix, _DEFAULT_CONTENT_TYPE)
    if not filename:
        filename = "audio.pcm" if is_pcm else "audio.wav"

    return filename, content_type


def _config_to_json(config: Any, *, exclude: Tuple[str, ...] = ()) -> Optional[dict]:
    """
    Serializes the config to the JSON `config` part.

    Args:
        config: the pydantic config model.
        exclude: field names to drop — options carried outside the body.

    Returns: the JSON-ready dict, or `None` when nothing is set.
    """
    data = config.dict(exclude_none=True)
    for field in exclude:
        data.pop(field, None)
    return data or None
