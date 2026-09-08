import json
from typing import Any, Callable, Dict, List, Optional, Union

from ... import client as _client
from . import api, models
from .params import LLMGatewayMessageParamUnion
from .stream import LLMGatewayStream


class ModelsResource:
    """The `models` resource of `LLMGateway`."""

    def __init__(self, gateway: "LLMGateway") -> None:
        self._gateway = gateway

    def list(self) -> models.LLMGatewayModelList:
        """
        Lists the models available through the LLM Gateway.

        Raises: `LLMGatewayError` if the request fails.
        """
        settings = self._gateway.client.settings

        return api.list_models(
            self._gateway.client.http_client,
            base_url=settings.llm_gateway_base_url,
            timeout=settings.llm_gateway_http_timeout,
        )


class CompletionsResource:
    """The `chat.completions` resource of `LLMGateway`."""

    def __init__(self, gateway: "LLMGateway") -> None:
        self._gateway = gateway

    def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParamUnion],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[models.LLMGatewayChatCompletion, LLMGatewayStream]:
        """
        Creates a chat completion.

        `messages` are plain dicts (see `LLMGatewayMessageParam`), e.g.
        `[{"role": "user", "content": "..."}]`. Extra keyword arguments —
        `max_tokens`, `temperature`, `tools`, `fallbacks`, `fallback_config`,
        `zero_data_retention`, `transcript_id`, `reasoning`, and any other
        field the API accepts — are forwarded directly into the request body.
        `fallbacks` is a list of dicts, each with the API's expected shape,
        e.g. `[{"model": "gpt-4o", "messages": [...]}]`.

        With `stream=True`, returns a `LLMGatewayStream` — iterate it for
        `LLMGatewayCompletionChunk`s exactly like the raw stream it replaces,
        then call `.get_final_message()` for the assembled text/usage — instead
        of a single `LLMGatewayChatCompletion`. Chunk parsing is verified for
        OpenAI-routed models only.

        Raises: `LLMGatewayError` if the request fails (for a stream, only if
            it fails before any bytes are received — a failure mid-stream
            just truncates the iterator).
        """
        settings = self._gateway.client.settings
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        if stream:
            return LLMGatewayStream(
                api.stream_completion(
                    self._gateway.client.http_client,
                    base_url=settings.llm_gateway_base_url,
                    timeout=settings.llm_gateway_http_timeout,
                    body=body,
                )
            )

        return api.create_completion(
            self._gateway.client.http_client,
            base_url=settings.llm_gateway_base_url,
            timeout=settings.llm_gateway_http_timeout,
            body=body,
        )

    def run_tools(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParamUnion],
        tools: List[Dict[str, Any]],
        functions: Dict[str, Callable[..., Any]],
        max_rounds: int = 10,
        **kwargs: Any,
    ) -> Union[models.LLMGatewayChatCompletion, LLMGatewayStream]:
        """
        Runs the tool-calling loop to completion: calls the model, invokes any
        requested tools from `functions`, feeds their results back, and repeats
        until the model responds without requesting a tool call.

        `messages` is mutated in place with every tool round-trip and the final
        assistant reply, so it's a complete transcript once this returns. Tool
        round-trips are replayed as `{"role": "assistant", "tool_calls": [...]}`
        followed by one `{"role": "tool", "tool_call_id": ..., "content": ...}`
        per call, per https://www.assemblyai.com/docs/llm-gateway/tool-calling.

        `functions` maps tool names (as declared in `tools`) to the Python
        callables that implement them; each is called with the model's parsed
        arguments as keyword arguments, and its return value is sent back to
        the model (JSON-encoded, unless it's already a string).

        Raises:
            ValueError: if `stream=True` is passed (unsupported), or the model
                requests a tool with no matching entry in `functions`.
            RuntimeError: if the loop doesn't finish within `max_rounds`.
            LLMGatewayError: if any underlying `create()` call fails.
        """
        if kwargs.get("stream"):
            raise ValueError("run_tools() does not support stream=True")

        for _ in range(max_rounds):
            completion = self.create(
                model=model, messages=messages, tools=tools, **kwargs
            )
            message = completion.choices[0].message

            if not message.tool_calls:
                messages.append({"role": "assistant", "content": message.content})
                return completion

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

        raise RuntimeError(f"Tool loop did not finish within max_rounds={max_rounds}")


class ChatResource:
    """The `chat` resource of `LLMGateway`."""

    def __init__(self, gateway: "LLMGateway") -> None:
        self.completions = CompletionsResource(gateway)


class UnderstandingResource:
    """The `understanding` resource of `LLMGateway`."""

    def __init__(self, gateway: "LLMGateway") -> None:
        self._gateway = gateway

    def create(
        self,
        *,
        transcript_id: str,
        request: Dict[str, Any],
        **kwargs: Any,
    ) -> models.LLMGatewayUnderstandingResponse:
        """
        Runs speech understanding features against an existing transcript.

        `request` is the `speech_understanding.request` feature config, e.g.
        `{"speaker_identification": {"speaker_type": "name"}}` — a plain
        dict, since the server itself has no fixed schema for it.

        Raises: `LLMGatewayError` if the request fails.
        """
        settings = self._gateway.client.settings
        body: Dict[str, Any] = {
            "transcript_id": transcript_id,
            "speech_understanding": {"request": request},
            **kwargs,
        }

        return api.create_understanding(
            self._gateway.client.http_client,
            base_url=settings.llm_gateway_base_url,
            timeout=settings.llm_gateway_http_timeout,
            body=body,
        )

    def validate(
        self,
        *,
        request: Dict[str, Any],
        transcript_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Validates a speech understanding request without running it.

        Returns `None` if the request is valid. Note this still requires
        the same authorization/balance as a real `create()` call — it is
        not a free dry run.

        Raises: `LLMGatewayError` (with `.errors` populated with the
            validation messages) if the request is invalid.
        """
        settings = self._gateway.client.settings
        body: Dict[str, Any] = {"speech_understanding": {"request": request}, **kwargs}
        if transcript_id is not None:
            body["transcript_id"] = transcript_id

        api.validate_understanding(
            self._gateway.client.http_client,
            base_url=settings.llm_gateway_base_url,
            timeout=settings.llm_gateway_http_timeout,
            body=body,
        )


class LLMGateway:
    """
    Client for AssemblyAI's LLM Gateway.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        gateway = aai.LLMGateway()
        for model in gateway.models.list().data:
            print(model.id)

        completion = gateway.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Summarize this call."}],
        )
        print(completion.choices[0].message.content)
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates an `LLMGateway`.

        Args:
            client: The HTTP client to use. Defaults to the shared default client.
            api_key: The API key to authenticate with. Builds a `Client` for this
                gateway. Given alongside `client`, it takes precedence: the
                gateway builds its own client from a copy of that client's
                settings with the key replaced, and the given client is left
                untouched.
        """
        self._client = _client._resolve_client(client, api_key)
        self.models = ModelsResource(self)
        self.chat = ChatResource(self)
        self.understanding = UnderstandingResource(self)

    @property
    def client(self) -> _client.Client:
        """The `Client` this gateway sends requests with."""

        return self._client
