"""The Text-to-Speech API: text in, audio out, one text per HTTP request.

The implementation lives in ``tts/v1/``, mirroring the ``dictation/v1/`` and
``sync/v1/`` layout. This ``__init__`` re-exports the public surface so
``from assemblyai.tts import TextToSpeech`` works alongside the versioned path.
"""

from .v1 import (
    AsyncTextToSpeech,
    AsyncTTSSession,
    AsyncTTSStream,
    TextToSpeech,
    TTSAudioEvent,
    TTSBeginEvent,
    TTSCancelledEvent,
    TTSConfig,
    TTSContainer,
    TTSEncoding,
    TTSError,
    TTSEvents,
    TTSFlushDoneEvent,
    TTSResponse,
    TTSResponseInfo,
    TTSSession,
    TTSStream,
    TTSTerminationEvent,
    TTSWord,
    TTSWordBoundariesEvent,
)

__all__ = [
    "AsyncTTSSession",
    "AsyncTTSStream",
    "AsyncTextToSpeech",
    "TTSAudioEvent",
    "TTSBeginEvent",
    "TTSCancelledEvent",
    "TTSConfig",
    "TTSEvents",
    "TTSFlushDoneEvent",
    "TTSSession",
    "TTSTerminationEvent",
    "TTSWord",
    "TTSWordBoundariesEvent",
    "TTSContainer",
    "TTSEncoding",
    "TTSError",
    "TTSResponse",
    "TTSResponseInfo",
    "TTSStream",
    "TextToSpeech",
]
