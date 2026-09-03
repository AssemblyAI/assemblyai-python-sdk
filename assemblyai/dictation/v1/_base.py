from __future__ import annotations

from typing import Mapping, Optional, Tuple

from ... import _audio
from ... import client as _client
from ..._audio import AudioInput
from . import api
from .models import DictationConfig, DictationResponse

__all__ = [
    "AudioInput",
    "_DictationTranscriberImpl",
    "_config_to_json",
    "_resolve_audio",
    "check_config",
]

# The Dictation API decodes raw PCM and the common container formats; the
# extension picks the Content-Type that tells the server which decoder to use.
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


def _resolve_audio(
    data: AudioInput,
    config: DictationConfig,
    owner: str,
) -> Tuple[bytes, str, str]:
    """
    Reads the audio input into bytes and decides its multipart Content-Type.

    PCM is selected when the source has a `.pcm`/`.raw` extension or when
    `sample_rate`/`channels` are set on the config (the fields the Dictation
    API requires only for raw PCM) — and both must then be present. Any other
    extension maps to its container Content-Type, and audio with no usable
    extension is posted as WAV. URLs are rejected — the Dictation API has no
    URL ingestion.

    Args:
        data: a local file path, raw bytes, or a binary file object.
        config: the configuration for this call.
        owner: the class to name in error messages, so each client reports
            its own name.

    Returns: `(audio_bytes, filename, content_type)`.
    """

    return _audio._resolve_audio(
        data,
        sample_rate=config.sample_rate,
        channels=config.channels,
        owner=owner,
        config_name="DictationConfig",
        content_types=_CONTENT_TYPES,
    )


def _config_to_json(config: DictationConfig) -> Optional[dict]:
    """Serializes the config to the JSON `config` part."""

    return _audio._config_to_json(config)


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

    def transcribe(
        self,
        *,
        data: AudioInput,
        config: Optional[DictationConfig],
    ) -> DictationResponse:
        config = config or self.config
        audio, filename, content_type = _resolve_audio(data, config, self._owner)
        return api.transcribe(
            self._client.http_client,
            base_url=self._client.settings.dictation_base_url,
            audio=audio,
            filename=filename,
            audio_content_type=content_type,
            config=_config_to_json(config),
            timeout=self._client.settings.dictation_http_timeout,
        )
