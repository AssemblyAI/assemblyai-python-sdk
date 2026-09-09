"""The asyncio counterpart of `api.py`.

Calls the same endpoints as their sync twins and raises the same
`LLMGatewayError` through `api._error_from_response`.
"""

import json
from typing import Any, AsyncIterator, Dict

import httpx

from . import models
from .api import (
    ENDPOINT_CHAT_COMPLETIONS,
    ENDPOINT_MODELS,
    ENDPOINT_UNDERSTANDING,
    ENDPOINT_UNDERSTANDING_VALIDATE,
    _error_from_response,
)

__all__ = [
    "list_models",
    "create_completion",
    "stream_completion",
    "create_understanding",
    "validate_understanding",
]


async def list_models(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout: float,
) -> models.LLMGatewayModelList:
    """
    Lists the models available through the LLM Gateway.

    Raises: `LLMGatewayError` on any non-200 response.
    """
    response = await client.get(
        base_url.rstrip("/") + ENDPOINT_MODELS,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayModelList.parse_obj(response.json())


async def create_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> models.LLMGatewayChatCompletion:
    """
    Posts a non-streaming chat completion request.

    Raises: `LLMGatewayError` on any non-200 response.
    """
    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_CHAT_COMPLETIONS,
        json=body,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayChatCompletion.parse_obj(response.json())


async def stream_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> AsyncIterator[models.LLMGatewayCompletionChunk]:
    """
    Posts a streaming chat completion request and yields parsed chunks.

    Chunk parsing is verified against OpenAI-routed models only (`data: {...}`
    SSE lines terminated by `data: [DONE]`); other providers' streaming
    framing was not verified and may not parse into `LLMGatewayCompletionChunk`.

    Raises: `LLMGatewayError` if the request fails before any bytes stream.
    """
    async with client.stream(
        "POST",
        base_url.rstrip("/") + ENDPOINT_CHAT_COMPLETIONS,
        json=body,
        timeout=timeout,
    ) as response:
        if response.status_code != httpx.codes.OK:
            await response.aread()
            raise _error_from_response(response)

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                break
            yield models.LLMGatewayCompletionChunk.parse_obj(json.loads(payload))


async def create_understanding(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> models.LLMGatewayUnderstandingResponse:
    """
    Runs speech understanding features against an existing transcript.

    Raises: `LLMGatewayError` on any non-200 response.
    """
    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_UNDERSTANDING,
        json=body,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayUnderstandingResponse.parse_obj(response.json())


async def validate_understanding(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> None:
    """
    Validates a speech understanding request without running it.

    Raises: `LLMGatewayError` (with `.errors` populated) if the request is invalid.
    """
    response = await client.post(
        base_url.rstrip("/") + ENDPOINT_UNDERSTANDING_VALIDATE,
        json=body,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)
