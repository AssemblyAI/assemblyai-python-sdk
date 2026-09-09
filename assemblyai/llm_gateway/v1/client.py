from typing import Any, Dict, Iterator, List, Literal, Optional, Union, overload

from ... import client as _client
from . import _base, api, models
from .params import LLMGatewayMessageParam


class ModelsResource(_base._BaseResource):
    """The `models` resource of `LLMGateway`."""

    def list(self) -> models.LLMGatewayModelList:
        """
        Lists the models available through the LLM Gateway.

        Raises: `LLMGatewayError` if the request fails.
        """
        return api.list_models(
            self._gateway.client.http_client,
            **self._http_options,
        )


class CompletionsResource(_base._BaseResource):
    """The `chat.completions` resource of `LLMGateway`."""

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> models.LLMGatewayChatCompletion: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: Literal[True],
        **kwargs: Any,
    ) -> Iterator[models.LLMGatewayCompletionChunk]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: bool,
        **kwargs: Any,
    ) -> Union[
        models.LLMGatewayChatCompletion, Iterator[models.LLMGatewayCompletionChunk]
    ]: ...

    def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        models.LLMGatewayChatCompletion, Iterator[models.LLMGatewayCompletionChunk]
    ]:
        """
        Creates a chat completion.

        `messages` are plain dicts (see `LLMGatewayMessageParam`), e.g.
        `[{"role": "user", "content": "..."}]`. Extra keyword arguments —
        `max_tokens`, `temperature`, `tools`, `fallbacks`, `fallback_config`,
        `zero_data_retention`, `transcript_id`, `reasoning`, and any other
        field the API accepts — are forwarded directly into the request body.
        `fallbacks` is a list of dicts, each with the API's expected shape,
        e.g. `[{"model": "gpt-4o", "messages": [...]}]`.

        With `stream=True`, returns an iterator of `LLMGatewayCompletionChunk`
        instead of a single `LLMGatewayChatCompletion`. Chunk parsing is
        verified for OpenAI-routed models only.

        Raises: `LLMGatewayError` if the request fails (for a stream, only if
            it fails before any bytes are received — a failure mid-stream
            just truncates the iterator).
        """
        body = _base.completion_body(
            model=model, messages=messages, stream=stream, extra=kwargs
        )

        if stream:
            return api.stream_completion(
                self._gateway.client.http_client,
                body=body,
                **self._http_options,
            )

        return api.create_completion(
            self._gateway.client.http_client,
            body=body,
            **self._http_options,
        )


class ChatResource:
    """The `chat` resource of `LLMGateway`."""

    def __init__(self, gateway: "LLMGateway") -> None:
        self.completions = CompletionsResource(gateway)


class UnderstandingResource(_base._BaseResource):
    """The `understanding` resource of `LLMGateway`."""

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
        return api.create_understanding(
            self._gateway.client.http_client,
            body=_base.understanding_body(
                transcript_id=transcript_id, request=request, extra=kwargs
            ),
            **self._http_options,
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
        api.validate_understanding(
            self._gateway.client.http_client,
            body=_base.validate_body(
                request=request, transcript_id=transcript_id, extra=kwargs
            ),
            **self._http_options,
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
