import json
from typing import Any, Dict, Iterator, List, Optional

import httpx

from ... import types
from . import models

ENDPOINT_MODELS = "/v1/models"
ENDPOINT_CHAT_COMPLETIONS = "/v1/chat/completions"
ENDPOINT_UNDERSTANDING = "/v1/understanding"
ENDPOINT_UNDERSTANDING_VALIDATE = "/v1/understanding/validate"


def _error_from_response(response: httpx.Response) -> types.LLMGatewayError:
    """
    Builds an `LLMGatewayError` from a non-200 response.

    The service uses two envelopes: business errors
    (`{"message", "code", "request_id", "metadata": {"errors": [...]}}`, from
    route handlers) and 401s raised by auth middleware before a request
    reaches a handler (`{"error", "status", "request_id"}`). Both are
    detected here. A missing or unparsable body falls back to the raw
    response text or a generic message.
    """
    request_id: Optional[str] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None

    print(response.text)

    try:
        body = response.json()
        if isinstance(body, dict):
            request_id = body.get("request_id")
            if "message" in body:
                message = body.get("message")
                metadata = body.get("metadata")
                if isinstance(metadata, dict) and isinstance(metadata.get("errors"), list):
                    errors = metadata["errors"]
            elif "error" in body:
                message = body.get("error")
    except Exception:
        message = response.text or None

    if not message:
        message = f"LLM Gateway request failed with status {response.status_code}"

    return types.LLMGatewayError(
        message,
        status_code=response.status_code,
        request_id=request_id,
        errors=errors,
    )


def list_models(
    client: httpx.Client,
    *,
    base_url: str,
    timeout: float,
) -> models.LLMGatewayModelList:
    """
    Lists the models available through the LLM Gateway.

    Args:
        client: the HTTP client (carries the `authorization` header, though
            this endpoint does not require one).
        base_url: the LLM Gateway base URL, e.g. `https://llm-gateway.assemblyai.com`.
        timeout: per-request timeout in seconds.

    Raises: `LLMGatewayError` on any non-200 response.
    """
    response = client.get(
        base_url.rstrip("/") + ENDPOINT_MODELS,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayModelList.parse_obj(response.json())


def create_completion(
    client: httpx.Client,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> models.LLMGatewayChatCompletion:
    """
    Posts a non-streaming chat completion request.

    Raises: `LLMGatewayError` on any non-200 response.
    """

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_CHAT_COMPLETIONS,
        json=body,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayChatCompletion.parse_obj(response.json())


def stream_completion(
    client: httpx.Client,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> Iterator[models.LLMGatewayCompletionChunk]:
    """
    Posts a streaming chat completion request and yields parsed chunks.

    Chunk parsing is verified against OpenAI-routed models only (`data: {...}`
    SSE lines terminated by `data: [DONE]`); other providers' streaming
    framing was not verified and may not parse into `LLMGatewayCompletionChunk`.

    Raises: `LLMGatewayError` if the request fails before any bytes stream.
    """
    with client.stream(
        "POST",
        base_url.rstrip("/") + ENDPOINT_CHAT_COMPLETIONS,
        json=body,
        timeout=timeout,
    ) as response:
        if response.status_code != httpx.codes.OK:
            response.read()
            raise _error_from_response(response)

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                break
            yield models.LLMGatewayCompletionChunk.parse_obj(json.loads(payload))


def create_understanding(
    client: httpx.Client,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> models.LLMGatewayUnderstandingResponse:
    """
    Runs speech understanding features against an existing transcript.

    Raises: `LLMGatewayError` on any non-200 response.
    """
    response = client.post(
        base_url.rstrip("/") + ENDPOINT_UNDERSTANDING,
        json=body,
        timeout=timeout,
    )

    from pprint import pprint
    pprint (response.json())

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return models.LLMGatewayUnderstandingResponse.parse_obj(response.json())


def validate_understanding(
    client: httpx.Client,
    *,
    base_url: str,
    timeout: float,
    body: Dict[str, Any],
) -> None:
    """
    Validates a speech understanding request without running it.

    Raises: `LLMGatewayError` (with `.errors` populated) if the request is invalid.
    """
    response = client.post(
        base_url.rstrip("/") + ENDPOINT_UNDERSTANDING_VALIDATE,
        json=body,
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)
