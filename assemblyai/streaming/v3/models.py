from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel


class LLMGatewayMessage(BaseModel):
    role: str
    content: str


class LLMGatewayConfig(BaseModel):
    model: str
    messages: List["LLMGatewayMessage"]
    max_tokens: int


class Word(BaseModel):
    start: int
    end: int
    confidence: float
    text: str
    word_is_final: bool
    speaker: Optional[str] = None


class TurnEvent(BaseModel):
    type: Literal["Turn"]
    turn_order: int
    turn_is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: List[Word]
    language_code: Optional[str] = None
    language_confidence: Optional[float] = None
    speaker_label: Optional[str] = None


class BeginEvent(BaseModel):
    type: Literal["Begin"] = "Begin"
    id: str
    expires_at: datetime


class TerminationEvent(BaseModel):
    type: Literal["Termination"] = "Termination"
    audio_duration_seconds: Optional[int] = None
    session_duration_seconds: Optional[int] = None


class SpeechStartedEvent(BaseModel):
    type: Literal["SpeechStarted"] = "SpeechStarted"
    timestamp: int


class ErrorEvent(BaseModel):
    type: Literal["Error"] = "Error"
    error_code: Optional[int] = None
    error: str


class WarningEvent(BaseModel):
    type: Literal["Warning"] = "Warning"
    warning_code: int
    warning: str


class LLMGatewayResponseEvent(BaseModel):
    type: Literal["LLMGatewayResponse"] = "LLMGatewayResponse"
    turn_order: int
    transcript: str
    data: Any


EventMessage = Union[
    BeginEvent,
    TerminationEvent,
    TurnEvent,
    SpeechStartedEvent,
    ErrorEvent,
    WarningEvent,
    LLMGatewayResponseEvent,
]


class TerminateSession(BaseModel):
    type: Literal["Terminate"] = "Terminate"


class ForceEndpoint(BaseModel):
    type: Literal["ForceEndpoint"] = "ForceEndpoint"


class StreamingSessionParameters(BaseModel):
    end_of_turn_confidence_threshold: Optional[float] = None
    min_end_of_turn_silence_when_confident: Optional[int] = (
        None  # Deprecated: Use min_turn_silence instead
    )
    min_turn_silence: Optional[int] = None
    max_turn_silence: Optional[int] = None
    vad_threshold: Optional[float] = None
    format_turns: Optional[bool] = None
    keyterms_prompt: Optional[List[str]] = None
    filter_profanity: Optional[bool] = None
    prompt: Optional[str] = None


class Encoding(str, Enum):
    pcm_s16le = "pcm_s16le"
    pcm_mulaw = "pcm_mulaw"

    def __str__(self):
        return self.value


class SpeechModel(str, Enum):
    universal_streaming_multilingual = "universal-streaming-multilingual"
    universal_streaming_english = "universal-streaming-english"
    u3_rt_pro = "u3-rt-pro"
    whisper_rt = "whisper-rt"
    u3_pro = "u3-pro"  # Deprecated: Use u3_rt_pro instead

    def __str__(self):
        return self.value


class StreamingDomain(str, Enum):
    medical = "medical-v1"

    def __str__(self):
        return self.value


class NoiseSuppressionModel(str, Enum):
    near_field = "near-field"
    far_field = "far-field"

    def __str__(self):
        return self.value


class StreamingPiiSubstitution(str, Enum):
    hash = "hash"
    entity_name = "entity_name"

    def __str__(self):
        return self.value


class StreamingPiiPolicy(str, Enum):
    account_number = "account_number"
    banking_information = "banking_information"
    blood_type = "blood_type"
    credit_card_number = "credit_card_number"
    credit_card_expiration = "credit_card_expiration"
    credit_card_cvv = "credit_card_cvv"
    date = "date"
    date_interval = "date_interval"
    date_of_birth = "date_of_birth"
    drivers_license = "drivers_license"
    drug = "drug"
    duration = "duration"
    email_address = "email_address"
    event = "event"
    filename = "filename"
    gender_sexuality = "gender_sexuality"
    gender = "gender"
    healthcare_number = "healthcare_number"
    injury = "injury"
    ip_address = "ip_address"
    language = "language"
    location = "location"
    marital_status = "marital_status"
    medical_condition = "medical_condition"
    medical_process = "medical_process"
    money_amount = "money_amount"
    nationality = "nationality"
    number_sequence = "number_sequence"
    passport_number = "passport_number"
    password = "password"
    person_age = "person_age"
    person_name = "person_name"
    phone_number = "phone_number"
    physical_attribute = "physical_attribute"
    political_affiliation = "political_affiliation"
    occupation = "occupation"
    organization = "organization"
    organization_medical_facility = "organization_medical_facility"
    religion = "religion"
    sexuality = "sexuality"
    statistics = "statistics"
    time = "time"
    url = "url"
    us_social_security_number = "us_social_security_number"
    username = "username"
    vehicle_id = "vehicle_id"
    zodiac_sign = "zodiac_sign"

    def __str__(self):
        return self.value


class StreamingParameters(StreamingSessionParameters):
    sample_rate: int
    encoding: Optional[Encoding] = None
    speech_model: SpeechModel
    language_detection: Optional[bool] = None
    domain: Optional[StreamingDomain] = None
    inactivity_timeout: Optional[int] = None
    webhook_url: Optional[str] = None
    webhook_auth_header_name: Optional[str] = None
    webhook_auth_header_value: Optional[str] = None
    llm_gateway: Optional[LLMGatewayConfig] = None
    speaker_labels: Optional[bool] = None
    max_speakers: Optional[int] = None
    voice_focus: Optional[NoiseSuppressionModel] = None
    voice_focus_threshold: Optional[float] = None
    # Deprecated: use voice_focus / voice_focus_threshold instead.
    noise_suppression_model: Optional[NoiseSuppressionModel] = None
    noise_suppression_threshold: Optional[float] = None
    continuous_partials: Optional[bool] = None
    customer_support_audio_capture: Optional[bool] = None
    include_partial_turns: Optional[bool] = None
    redact_pii: Optional[bool] = None
    redact_pii_policies: Optional[List[StreamingPiiPolicy]] = None
    redact_pii_sub: Optional[StreamingPiiSubstitution] = None


class UpdateConfiguration(StreamingSessionParameters):
    type: Literal["UpdateConfiguration"] = "UpdateConfiguration"


OperationMessage = Union[
    bytes,
    TerminateSession,
    ForceEndpoint,
    UpdateConfiguration,
]


class StreamingClientOptions(BaseModel):
    api_host: str = "streaming.assemblyai.com"
    api_key: Optional[str] = None
    token: Optional[str] = None


class StreamingError(Exception):
    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


StreamingErrorCodes = {
    3005: "Server error",
    3006: "Input validation error",
    3007: "Audio chunk duration violation",
    3008: "Session expired: maximum session duration exceeded",
    3009: "Too many concurrent sessions",
    4000: "Sample rate must be a positive integer",
    4001: "Not Authorized",
    4002: "Insufficient Funds",
    4003: """This feature is paid-only and requires you to add a credit card.
    Please visit https://app.assemblyai.com/ to add a credit card to your account""",
    4004: "Session Not Found",
    4008: "Session Expired",
    4010: "Session Previously Closed",
    4029: "Client sent audio too fast",
    4030: "Session is handled by another websocket",
    4031: "Session idle for too long",
    4032: "Audio duration is too short",
    4033: "Audio duration is too long",
    4034: "Audio too small to transcode",
    4100: "Endpoint received invalid JSON",
    4101: "Endpoint received a message with an invalid schema",
    4102: "This account has exceeded the number of allowed streams",
    4103: "The session has been reconnected. This websocket is no longer valid.",
    1013: "Temporary server condition forced blocking client's request",
}


class StreamingEvents(Enum):
    Begin = "Begin"
    Termination = "Termination"
    Turn = "Turn"
    SpeechStarted = "SpeechStarted"
    Error = "Error"
    Warning = "Warning"
    LLMGatewayResponse = "LLMGatewayResponse"
