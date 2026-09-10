"""The Dictation API: audio in as it is spoken, transcript (and optional LLM
rewrite) out.

The implementation lives in ``dictation/v1/``, mirroring the ``sync/v1/`` and
``streaming/v3/`` layout. This ``__init__`` re-exports the public surface so
``from assemblyai.dictation import DictationTranscriber`` works alongside the
versioned path.
"""

from .v1 import (
    AsyncAudioSource,
    AsyncDictationLiveSession,
    AsyncDictationTranscriber,
    AudioSource,
    DictationConfig,
    DictationError,
    DictationLiveSession,
    DictationResponse,
    DictationTranscriber,
    DictationWord,
)

__all__ = [
    "AsyncAudioSource",
    "AsyncDictationLiveSession",
    "AsyncDictationTranscriber",
    "AudioSource",
    "DictationConfig",
    "DictationError",
    "DictationLiveSession",
    "DictationResponse",
    "DictationTranscriber",
    "DictationWord",
]
