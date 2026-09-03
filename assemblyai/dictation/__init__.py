"""The Dictation API: audio in, transcript (and optional LLM rewrite) out.

The implementation lives in ``dictation/v1/``, mirroring the ``sync/v1/`` and
``streaming/v3/`` layout. This ``__init__`` re-exports the public surface so
``from assemblyai.dictation import DictationTranscriber`` works alongside the
versioned path.
"""

from .v1 import (
    AsyncDictationTranscriber,
    AudioInput,
    DictationConfig,
    DictationError,
    DictationResponse,
    DictationTranscriber,
    DictationWord,
)

__all__ = [
    "AsyncDictationTranscriber",
    "AudioInput",
    "DictationConfig",
    "DictationError",
    "DictationResponse",
    "DictationTranscriber",
    "DictationWord",
]
