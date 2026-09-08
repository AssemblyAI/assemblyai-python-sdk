"""Logic shared by `client.py` and `async_client.py`.

Holds the request body shapes and the tool-calling loop's core, so a
wire-contract or tool-loop change lands in one place. Everything here is
I/O-free — the sync and async clients supply the HTTP calls themselves.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from . import models
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


def reject_stream(kwargs: Dict[str, Any]) -> None:
    """Raises `ValueError` if `run_tools()` was passed `stream=True`."""
    if kwargs.get("stream"):
        raise ValueError("run_tools() does not support stream=True")


def apply_tool_round(
    completion: models.LLMGatewayChatCompletion,
    messages: List[LLMGatewayMessageParam],
    functions: Dict[str, Callable[..., Any]],
) -> bool:
    """
    Plays one round of the tool-calling loop into `messages`, in place.

    Appends the model's reply, and — when it requested tool calls — invokes each
    one from `functions` and appends its result.

    Returns: True once the model has replied without requesting a tool call,
        i.e. the loop is finished.

    Raises:
        ValueError: if the model requests a tool with no matching entry in
            `functions`.
    """
    message = completion.choices[0].message

    if not message.tool_calls:
        messages.append({"role": "assistant", "content": message.content})
        return True

    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                        if isinstance(tool_call.function.arguments, str)
                        else json.dumps(tool_call.function.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
    )

    for tool_call in message.tool_calls:
        function = functions.get(tool_call.function.name)
        if function is None:
            raise ValueError(
                f"No function registered for tool call {tool_call.function.name!r}"
            )

        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        result = function(**arguments)
        output = result if isinstance(result, str) else json.dumps(result)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
                "name": tool_call.function.name,
            }
        )

    return False
