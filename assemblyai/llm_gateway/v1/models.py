from typing import Any, Dict, List, Optional

from ...types import BaseModel, ConfigDict, Field, pydantic_v2


class LLMGatewayPricingData(BaseModel):
    completions: float
    prompt: float
    input_cache_read: Optional[float] = None
    input_cache_write: Optional[float] = None
    input_cache_write_1h: Optional[float] = None


class LLMGatewayPricing(BaseModel):
    us: Optional[LLMGatewayPricingData] = None
    eu: Optional[LLMGatewayPricingData] = None
    global_: LLMGatewayPricingData = Field(alias="global")
    regional_increase_percent: Optional[float] = None

    if pydantic_v2:
        model_config = ConfigDict(populate_by_name=True)
    else:

        class Config:
            allow_population_by_field_name = True


class LLMGatewayDefaultParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[int] = None


class LLMGatewayTopProvider(BaseModel):
    is_moderated: bool
    context_length: int
    max_completion_tokens: int


class LLMGatewayModel(BaseModel):
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


class LLMGatewayModelList(BaseModel):
    """The response of `GET /v1/models`."""

    data: List[LLMGatewayModel] = Field(default_factory=list)


class LLMGatewayCacheCreation(BaseModel):
    ephemeral_5m_input_tokens: int
    ephemeral_1h_input_tokens: int


class LLMGatewayPromptTokensDetails(BaseModel):
    cached_tokens: int
    audio_tokens: int
    cache_creation: Optional[LLMGatewayCacheCreation] = None
    cache_write_tokens: Optional[int] = None


class LLMGatewayCompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class LLMGatewayUsage(BaseModel):
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
    prompt_tokens_details: LLMGatewayPromptTokensDetails
    completion_tokens_details: LLMGatewayCompletionTokensDetails


class LLMGatewayFunction(BaseModel):
    name: str
    arguments: Any = None


class LLMGatewayToolCall(BaseModel):
    id: str
    type: str
    function: LLMGatewayFunction


class LLMGatewayResponseMessage(BaseModel):
    role: str
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[List[LLMGatewayToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class LLMGatewayChoice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    message: LLMGatewayResponseMessage


class LLMGatewayChatCompletion(BaseModel):
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


class LLMGatewayChunkDelta(BaseModel):
    content: Optional[str] = None


class LLMGatewayChunkChoice(BaseModel):
    index: int
    delta: LLMGatewayChunkDelta
    finish_reason: Optional[str] = None


class LLMGatewayCompletionChunk(BaseModel):
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


class LLMGatewayUnderstandingResponse(BaseModel):
    """The response of `POST /v1/understanding`."""

    speech_understanding: Dict[str, Any] = Field(default_factory=dict)
    request_id: str
    utterances: Optional[List[Dict[str, Any]]] = None
    translated_texts: Optional[Dict[str, str]] = None
