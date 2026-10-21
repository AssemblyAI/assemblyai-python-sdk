"""The Text-to-Speech API: text in, audio out, one text per HTTP request.

The implementation lives in ``tts/v1/``, mirroring the ``dictation/v1/`` and
``sync/v1/`` layout. This ``__init__`` re-exports the public surface so
``from assemblyai.tts import TextToSpeech`` works alongside the versioned path.
"""

from .v1 import (
    AsyncTextToSpeech,
    AsyncTTSStream,
    TextToSpeech,
    TTSConfig,
    TTSContainer,
    TTSEncoding,
    TTSError,
    TTSResponse,
    TTSResponseInfo,
    TTSStream,
)

__all__ = [
    "AsyncTTSStream",
    "AsyncTextToSpeech",
    "TTSConfig",
    "TTSContainer",
    "TTSEncoding",
    "TTSError",
    "TTSResponse",
    "TTSResponseInfo",
    "TTSStream",
    "TextToSpeech",
]
