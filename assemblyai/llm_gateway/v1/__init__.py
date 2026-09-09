"""LLM Gateway client surface.

Exports the clients, the request param type, and the response types a caller
receives or annotates. The nested leaf models — token-detail breakdowns,
pricing, provider metadata, streamed tool-call fragments — are reached by
attribute access off these types and stay in `models` rather than the public
surface.
"""

from ...types import LLMGatewayError
from .async_client import AsyncLLMGateway
from .client import LLMGateway
from .models import (
    LLMGatewayChatCompletion,
    LLMGatewayChoice,
    LLMGatewayChunkChoice,
    LLMGatewayChunkDelta,
    LLMGatewayCompletionChunk,
    LLMGatewayModel,
    LLMGatewayModelList,
    LLMGatewayResponseMessage,
    LLMGatewayToolCall,
    LLMGatewayUnderstandingResponse,
    LLMGatewayUsage,
)
from .params import LLMGatewayMessageParam

__all__ = [
    "AsyncLLMGateway",
    "LLMGateway",
    "LLMGatewayChatCompletion",
    "LLMGatewayChoice",
    "LLMGatewayChunkChoice",
    "LLMGatewayChunkDelta",
    "LLMGatewayCompletionChunk",
    "LLMGatewayError",
    "LLMGatewayMessageParam",
    "LLMGatewayModel",
    "LLMGatewayModelList",
    "LLMGatewayResponseMessage",
    "LLMGatewayToolCall",
    "LLMGatewayUnderstandingResponse",
    "LLMGatewayUsage",
]
