from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ... import _audio, types
from ..._audio import AudioInput
from ...dictation.v1._base import _CONTENT_TYPES
from .models import DiarizationConfig

__all__ = [
    "AudioInput",
    "TranscriptInput",
    "check_config",
    "resolve_request",
]

# A transcript to label: the sync API's response (request it with
# `timestamps=True`), a dict with a `words` list, or the list of words alone.
# Each word is a `SyncWord` or a dict with `text`, `start` and `end` in
# milliseconds and an optional `confidence`; other fields are ignored.
TranscriptInput = Union[
    types.SyncTranscriptResponse,
    Mapping[str, Any],
    Sequence[Union[types.SyncWord, Mapping[str, Any]]],
]


def check_config(owner: str, config: Optional[DiarizationConfig]) -> None:
    """Raises unless `config` is a `DiarizationConfig` or `None`."""

    _audio.check_config(owner, config, DiarizationConfig)


def _dump(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)
    return model


def _transcript_json(transcript: Optional[TranscriptInput]) -> Optional[Dict[str, Any]]:
    """The JSON `transcript` part, or `None` to leave the part out."""

    if transcript is None:
        return None
    if isinstance(transcript, Mapping):
        return dict(transcript)
    words: Any = getattr(transcript, "words", transcript)
    if isinstance(words, (str, bytes, bytearray)) or not isinstance(words, Sequence):
        raise TypeError(
            "transcript must be a SyncTranscriptResponse, a dict with `words`, "
            f"or a list of words, not {type(transcript).__name__}"
        )
    return {"words": [_dump(w) for w in words]}


def resolve_request(
    owner: str,
    audio: AudioInput,
    transcript: Optional[TranscriptInput],
    config: Optional[DiarizationConfig],
    content_type: Optional[str],
) -> Tuple[bytes, str, str, Optional[dict], Optional[dict]]:
    """
    Reads the audio and builds the parts of a diarize request.

    The audio's Content-Type comes from `content_type` when given, else from
    the file extension (`.mp3` -> `audio/mpeg`, ...), else `audio/pcm` when
    `sample_rate`/`channels` are set, else `audio/wav`.

    Returns: `(audio_bytes, filename, content_type, transcript_json, config_json)`.
    """

    check_config(owner, config)
    config = config or DiarizationConfig()
    data, filename, inferred = _audio._resolve_audio(
        audio,
        sample_rate=config.sample_rate,
        channels=config.channels,
        owner=owner,
        config_name="DiarizationConfig",
        content_types=_CONTENT_TYPES,
    )
    return (
        data,
        filename,
        content_type or inferred,
        _transcript_json(transcript),
        _audio._config_to_json(config),
    )
