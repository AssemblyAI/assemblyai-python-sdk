from typing import Any, Dict, List, Optional

from ...types import BaseModel, ConfigDict, Field, pydantic_v2

if pydantic_v2:
    from ...types import field_validator
else:
    from ...types import validator


class _LLMGatewayResponseModel(BaseModel):
    """
    Base for the LLM Gateway response models.

    Keeps unknown server fields reachable instead of dropping them, so a gateway
    that adds a field doesn't need an SDK release before callers can read it.
    """

    if pydantic_v2:
        model_config = ConfigDict(extra="allow")
    else:

        class Config:
            extra = "allow"


class LLMGatewayPricingData(_LLMGatewayResponseModel):
    completions: float
    prompt: float
    input_cache_read: Optional[float] = None
    input_cache_write: Optional[float] = None
    input_cache_write_1h: Optional[float] = None


class LLMGatewayPricing(_LLMGatewayResponseModel):
    us: Optional[LLMGatewayPricingData] = None
    eu: Optional[LLMGatewayPricingData] = None
    global_: LLMGatewayPricingData = Field(alias="global")
    regional_increase_percent: Optional[float] = None

    if pydantic_v2:
        model_config = ConfigDict(populate_by_name=True)
    else:

        class Config:
            allow_population_by_field_name = True


class LLMGatewayDefaultParameters(_LLMGatewayResponseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[int] = None


class LLMGatewayTopProvider(_LLMGatewayResponseModel):
    is_moderated: bool
    context_length: int
    max_completion_tokens: int


class LLMGatewayModel(_LLMGatewayResponseModel):
    """A model available through the LLM Gateway, as returned by `GET /v1/models`."""

    id: str
    name: str
    description: str
    default_parameters: LLMGatewayDefaultParameters
    supported_parameters: List[str] = Field(default_factory=list)
    top_provider: LLMGatewayTopProvider
    context_length: int
    pricing: LLMGatewayPricing
    creator: str
    retirement_date: int
    "Unix timestamp of the model's retirement date, or 0 if not retiring"

    available_regions: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    default_provider: str

    # The gateway is Go: a nil slice marshals as `null`, not `[]`, and an
    # explicit null bypasses the field default.
    if pydantic_v2:

        @field_validator(
            "supported_parameters", "available_regions", "providers", mode="before"
        )
        def set_collection_default(cls, v):
            return [] if v is None else v

    else:

        @validator("supported_parameters", "available_regions", "providers", pre=True)
        def set_collection_default(cls, v):
            return [] if v is None else v


class LLMGatewayModelList(_LLMGatewayResponseModel):
    """The response of `GET /v1/models`."""

    data: List[LLMGatewayModel] = Field(default_factory=list)

    if pydantic_v2:

        @field_validator("data", mode="before")
        def set_collection_default(cls, v):
            return [] if v is None else v

    else:

        @validator("data", pre=True)
        def set_collection_default(cls, v):
            return [] if v is None else v


class LLMGatewayCacheCreation(_LLMGatewayResponseModel):
    ephemeral_5m_input_tokens: int
    ephemeral_1h_input_tokens: int


class LLMGatewayPromptTokensDetails(_LLMGatewayResponseModel):
    cached_tokens: int
    audio_tokens: int
    cache_creation: Optional[LLMGatewayCacheCreation] = None
    cache_write_tokens: Optional[int] = None


class LLMGatewayCompletionTokensDetails(_LLMGatewayResponseModel):
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class LLMGatewayUsage(_LLMGatewayResponseModel):
    """
    `input_tokens`/`output_tokens` are only populated for non-streaming
    responses — the gateway forwards the upstream provider's own streaming
    usage chunk unmodified, which carries `prompt_tokens`/`completion_tokens`
    only.
    """

    input_tokens: Optional[int] = None
    prompt_tokens: int
    output_tokens: Optional[int] = None
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[LLMGatewayPromptTokensDetails] = None
    completion_tokens_details: Optional[LLMGatewayCompletionTokensDetails] = None


class LLMGatewayFunction(_LLMGatewayResponseModel):
    name: str
    arguments: Any = None


class LLMGatewayToolCall(_LLMGatewayResponseModel):
    id: str
    type: str
    function: LLMGatewayFunction


class LLMGatewayResponseMessage(_LLMGatewayResponseModel):
    role: str
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[List[LLMGatewayToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class LLMGatewayChoice(_LLMGatewayResponseModel):
    index: int
    finish_reason: Optional[str] = None
    message: LLMGatewayResponseMessage


class LLMGatewayChatCompletion(_LLMGatewayResponseModel):
    """The response of `POST /v1/chat/completions` (non-streaming)."""

    request_id: str
    choices: List[LLMGatewayChoice] = Field(default_factory=list)
    usage: LLMGatewayUsage
    request: Any = None
    "Loosely-typed echo of the request, as sent back by the server"

    http_status_code: Optional[int] = None
    response_time: Optional[int] = None
    "Nanoseconds, as sent by the server — not converted to seconds"

    llm_status_code: Optional[int] = None

    if pydantic_v2:

        @field_validator("choices", mode="before")
        def set_collection_default(cls, v):
            return [] if v is None else v

    else:

        @validator("choices", pre=True)
        def set_collection_default(cls, v):
            return [] if v is None else v


class LLMGatewayChunkFunction(_LLMGatewayResponseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None
    """A fragment of the JSON arguments, not the whole value — concatenate the
    fragments carrying the same `index` across chunks."""


class LLMGatewayChunkToolCall(_LLMGatewayResponseModel):
    """
    A tool call as it arrives mid-stream.

    Everything but `index` is optional: providers send the `id`/`type`/`name`
    once and then stream `function.arguments` in fragments, so a single chunk
    carries only part of the call.
    """

    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[LLMGatewayChunkFunction] = None


class LLMGatewayChunkDelta(_LLMGatewayResponseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[LLMGatewayChunkToolCall]] = None


class LLMGatewayChunkChoice(_LLMGatewayResponseModel):
    index: int
    delta: LLMGatewayChunkDelta
    finish_reason: Optional[str] = None


class LLMGatewayCompletionChunk(_LLMGatewayResponseModel):
    """
    A streamed chat completion chunk.

    Verified against OpenAI-routed models only — Claude/Gemini/Bedrock
    streaming framing was not verified and may not parse into this shape.
    """

    id: str
    object: str
    created: int
    model: str
    service_tier: Optional[str] = None
    system_fingerprint: Any = None
    choices: List[LLMGatewayChunkChoice] = Field(default_factory=list)
    usage: Optional[LLMGatewayUsage] = None
    obfuscation: Optional[str] = None

    if pydantic_v2:

        @field_validator("choices", mode="before")
        def set_collection_default(cls, v):
            return [] if v is None else v

    else:

        @validator("choices", pre=True)
        def set_collection_default(cls, v):
            return [] if v is None else v


class LLMGatewayUnderstandingResponse(_LLMGatewayResponseModel):
    """The response of `POST /v1/understanding`."""

    speech_understanding: Dict[str, Any] = Field(default_factory=dict)
    request_id: str
    utterances: Optional[List[Dict[str, Any]]] = None
    translated_texts: Optional[Dict[str, str]] = None

    if pydantic_v2:

        @field_validator("speech_understanding", mode="before")
        def set_collection_default(cls, v):
            return {} if v is None else v

    else:

        @validator("speech_understanding", pre=True)
        def set_collection_default(cls, v):
            return {} if v is None else v
