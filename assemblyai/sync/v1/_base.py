from __future__ import annotations

from typing import Mapping, Optional, Tuple

from ... import _audio, types
from ... import client as _client
from ..._audio import _PCM_SUFFIXES, AudioInput
from . import api

__all__ = [
    "_PCM_SUFFIXES",
    "AudioInput",
    "_SyncTranscriberImpl",
    "_config_to_json",
    "_resolve_audio",
    "check_config",
]

# The sync API decodes a WAV container or raw PCM; every other extension is
# posted as WAV and left to the server to sniff.
_CONTENT_TYPES: Mapping[str, str] = {}


def check_config(owner: str, config: Optional[types.SyncTranscriptionConfig]) -> None:
    """
    Raises unless `config` is a `SyncTranscriptionConfig` or `None`.

    The job API's `TranscriptionConfig` is a different, non-interchangeable
    type, so passing it here is a mistake worth naming.

    Args:
        owner: the class to name in the message, e.g. `SyncTranscriber`.
        config: the configuration to check.
    """

    _audio.check_config(owner, config, types.SyncTranscriptionConfig)


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

    return _audio._resolve_audio(
        data,
        sample_rate=config.sample_rate,
        channels=config.channels,
        owner="SyncTranscriber",
        config_name="SyncTranscriptionConfig",
        content_types=_CONTENT_TYPES,
    )


def _config_to_json(config: types.SyncTranscriptionConfig) -> Optional[dict]:
    """Serializes the config to the JSON `config` part, dropping the routing model."""

    return _audio._config_to_json(config, exclude=("model",))


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
