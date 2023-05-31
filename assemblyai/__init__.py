from .client import Client
from .lemur import Lemur
from .transcriber import Transcriber, Transcript, TranscriptGroup
from .types import (
    AssemblyAIError,
    LanguageCode,
    LemurError,
    LemurModel,
    LemurQuestion,
    LemurQuestionResult,
    Paragraph,
    PIIRedactionPolicy,
    PIISubstitutionPolicy,
    RawTranscriptionConfig,
    Sentence,
    Settings,
    SummarizationModel,
    SummarizationType,
    Timestamp,
    TranscriptError,
    TranscriptionConfig,
    TranscriptStatus,
    Utterance,
    UtteranceWord,
    Word,
    WordBoost,
    WordSearchMatch,
)

settings = Settings()
"""Global settings object that applies to all classes that use the `Client` class."""


__all__ = [
    # types
    "AssemblyAIError",
    "Client",
    "LanguageCode",
    "Lemur",
    "LemurError",
    "LemurModel",
    "LemurQuestion",
    "LemurQuestionResult",
    "Sentence",
    "Settings",
    "SummarizationModel",
    "SummarizationType",
    "Timestamp",
    "Transcriber",
    "TranscriptionConfig",
    "Transcript",
    "TranscriptError",
    "TranscriptGroup",
    "TranscriptStatus",
    "Utterance",
    "UtteranceWord",
    "Paragraph",
    "PIISubstitutionPolicy",
    "PIIRedactionPolicy",
    "RawTranscriptionConfig",
    "Word",
    "WordBoost",
    "WordSearchMatch",
    # package globals
    "settings",
]
