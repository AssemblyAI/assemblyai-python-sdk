"""The Diarization API: audio (and optionally its transcript) in, speaker labels out.

The implementation lives in ``diarization/v1/``, mirroring the ``tts/v1/`` and
``dictation/v1/`` layout. This ``__init__`` re-exports the public surface so
``from assemblyai.diarization import Diarizer`` works alongside the versioned
path.
"""

from .v1 import (
    AsyncDiarizer,
    DiarizationConfig,
    DiarizationError,
    DiarizationResponse,
    DiarizationSegment,
    DiarizationUtterance,
    DiarizedWord,
    Diarizer,
    TranscriptInput,
)

__all__ = [
    "AsyncDiarizer",
    "DiarizationConfig",
    "DiarizationError",
    "DiarizationResponse",
    "DiarizationSegment",
    "DiarizationUtterance",
    "DiarizedWord",
    "Diarizer",
    "TranscriptInput",
]
