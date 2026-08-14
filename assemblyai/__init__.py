from .__version__ import __version__
from .async_client import AsyncClient
from .client import Client
from .prerecorded.v2 import AsyncTranscriber, AsyncTranscript
from .sync.v1 import AsyncSyncTranscriber, SyncTranscriber
from .transcriber import Transcriber, Transcript, TranscriptGroup
from .types import (
    AssemblyAIError,
    AutohighlightResponse,
    AutohighlightResult,
    Chapter,
    ContentSafetyLabel,
    ContentSafetyLabelResult,
    ContentSafetyResponse,
    ContentSafetySeverityScore,
    CustomFormattingRequest,
    CustomFormattingResponse,
    Entity,
    EntityType,
    IABLabelResult,
    IABResponse,
    IABResult,
    KeytermsPromptOptions,
    LanguageCode,
    LanguageDetectionOptions,
    ListTranscriptParameters,
    ListTranscriptResponse,
    PageDetails,
    Paragraph,
    PIIRedactedAudioMethod,
    PIIRedactedAudioQuality,
    PIIRedactionPolicy,
    PIISubstitutionPolicy,
    RawTranscriptionConfig,
    RedactPiiAudioOptions,
    Sentence,
    Sentiment,
    SentimentType,
    Settings,
    SpeakerIdentificationRequest,
    SpeakerIdentificationResponse,
    SpeakerOptions,
    SpeakerType,
    SpeechModel,
    SpeechUnderstandingFeatureRequests,
    SpeechUnderstandingFeatureResponses,
    SpeechUnderstandingRequest,
    SpeechUnderstandingResponse,
    StatusResult,
    SummarizationModel,
    SummarizationType,
    SyncSpeechModel,
    SyncTranscriptError,
    SyncTranscriptionConfig,
    SyncTranscriptResponse,
    SyncWord,
    Timestamp,
    TranscriptError,
    TranscriptionConfig,
    TranscriptItem,
    TranscriptMetadata,
    TranscriptStatus,
    TranslationRequest,
    TranslationResponse,
    Utterance,
    UtteranceWord,
    Word,
    WordBoost,
    WordSearchMatch,
)

settings = Settings()
"""Global settings object that applies to all classes that use the `Client` class."""


def __getattr__(name: str):
    """
    Resolves `assemblyai.streaming` on first access.

    Streaming pulls in `websockets`, so the subpackage is imported when it is
    asked for rather than at `import assemblyai` time. The import binds
    `streaming` as a real attribute, so this runs only once.
    """

    if name == "streaming":
        import importlib

        importlib.import_module(".streaming.v3", __name__)

        return importlib.import_module(".streaming", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # types
    "AssemblyAIError",
    "AsyncClient",
    "AsyncSyncTranscriber",
    "AsyncTranscriber",
    "AsyncTranscript",
    "AutohighlightResponse",
    "AutohighlightResult",
    "Chapter",
    "Client",
    "ContentSafetyLabel",
    "ContentSafetyLabelResult",
    "ContentSafetyResponse",
    "ContentSafetySeverityScore",
    "CustomFormattingRequest",
    "CustomFormattingResponse",
    "Entity",
    "EntityType",
    "IABLabelResult",
    "IABResponse",
    "IABResult",
    "KeytermsPromptOptions",
    "LanguageCode",
    "LanguageDetectionOptions",
    "ListTranscriptParameters",
    "ListTranscriptResponse",
    "PageDetails",
    "Sentence",
    "Sentiment",
    "SentimentType",
    "Settings",
    "SpeakerIdentificationRequest",
    "SpeakerIdentificationResponse",
    "SpeakerOptions",
    "SpeakerType",
    "SpeechModel",
    "SpeechUnderstandingFeatureRequests",
    "SpeechUnderstandingFeatureResponses",
    "SpeechUnderstandingRequest",
    "SpeechUnderstandingResponse",
    "StatusResult",
    "SummarizationModel",
    "SummarizationType",
    "SyncSpeechModel",
    "SyncTranscriber",
    "SyncTranscriptError",
    "SyncTranscriptionConfig",
    "SyncTranscriptResponse",
    "SyncWord",
    "Timestamp",
    "Transcriber",
    "TranscriptionConfig",
    "Transcript",
    "TranscriptError",
    "TranscriptMetadata",
    "TranscriptGroup",
    "TranscriptItem",
    "TranslationRequest",
    "TranslationResponse",
    "TranscriptStatus",
    "Utterance",
    "UtteranceWord",
    "Paragraph",
    "PIIRedactedAudioMethod",
    "PIIRedactedAudioQuality",
    "PIISubstitutionPolicy",
    "PIIRedactionPolicy",
    "RedactPiiAudioOptions",
    "RawTranscriptionConfig",
    "Word",
    "WordBoost",
    "WordSearchMatch",
    # subpackages
    "streaming",
    # package globals
    "settings",
    # version
    "__version__",
]
