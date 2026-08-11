"""Backwards-compatible re-exports of the prerecorded transcription surface.

The canonical location is ``assemblyai.prerecorded.v2``.
"""

from .prerecorded.v2.client import Transcriber, _TranscriberImpl  # noqa: F401
from .prerecorded.v2.transcript import Transcript, _TranscriptImpl  # noqa: F401
from .prerecorded.v2.transcript_group import (  # noqa: F401
    TranscriptGroup,
    _TranscriptGroupImpl,
)

__all__ = [
    "Transcriber",
    "Transcript",
    "TranscriptGroup",
]
