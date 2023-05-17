from .transcriber import Transcriber, Transcript, TranscriptGroup
from .client import Client
from .lemur import Lemur

from .types import (
    AssemblyAIError,
    Settings,
    TranscriptError,
    TranscriptStatus,
    TranscriptionConfig,
    Utterance,
    UtteranceWord,
    LanguageCode,
    Paragraph,
    Sentence,
    LemurModel,
    LemurError,
    LemurQuestion,
    LemurQuestionResult,
    SummarizationModel,
    PIISubstitutionPolicy,
    RawTranscriptionConfig,
    SummarizationType,
    WordBoost,
    WordSearchMatch,
    Word,
    Timestamp,
    PIIRedactionPolicy,
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
