"""Logic shared by `client.py` and `async_client.py`.

Holds the request body shapes, so a wire-contract change lands in one place.
Everything here is I/O-free — the sync and async clients supply the HTTP
calls themselves.
"""

from typing import Any, Dict, List, Optional

from .params import LLMGatewayMessageParam


class _BaseResource:
    """
    Shared base for the gateway's resource classes.

    Subclasses hold a gateway (sync or async) and implement the I/O.
    """

    def __init__(self, gateway: Any) -> None:
        self._gateway = gateway

    @property
    def _http_options(self) -> Dict[str, Any]:
        """The `base_url`/`timeout` keyword arguments every `api` call takes."""
        settings = self._gateway.client.settings

        return {
            "base_url": settings.llm_gateway_base_url,
            "timeout": settings.llm_gateway_http_timeout,
        }


def completion_body(
    *,
    model: str,
    messages: List[LLMGatewayMessageParam],
    stream: bool,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the request body of `POST /v1/chat/completions`."""
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        **extra,
    }


def understanding_body(
    *,
    transcript_id: str,
    request: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the request body of `POST /v1/understanding`."""
    return {
        "transcript_id": transcript_id,
        "speech_understanding": {"request": request},
        **extra,
    }


def validate_body(
    *,
    request: Dict[str, Any],
    transcript_id: Optional[str],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the request body of `POST /v1/understanding/validate`."""
    body: Dict[str, Any] = {"speech_understanding": {"request": request}, **extra}
    if transcript_id is not None:
        body["transcript_id"] = transcript_id

    return body
