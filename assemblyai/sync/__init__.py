"""Backwards-compatibility package for the old flat ``sync.py`` module.

The sync (single-request) transcription product now lives in ``sync/v1/``,
mirroring the ``streaming/v3/`` layout. This ``__init__`` re-exports the full
surface the old ``assemblyai.sync`` module exposed so every existing import —
``from assemblyai.sync import SyncTranscriber``, the ``AudioInput`` alias, and
the private helpers — keeps working silently.
"""

from .v1._base import (  # noqa: F401
    _PCM_SUFFIXES,
    AudioInput,
    _config_to_json,
    _resolve_audio,
    _SyncTranscriberImpl,
)
from .v1.client import SyncTranscriber

__all__ = [
    "AudioInput",
    "SyncTranscriber",
]
