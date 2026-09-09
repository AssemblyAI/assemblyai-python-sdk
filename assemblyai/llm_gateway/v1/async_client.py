from __future__ import annotations

from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    overload,
)

from typing_extensions import Self

from ... import async_client as _async_client
from . import _base, async_api, models
from .params import LLMGatewayMessageParam


class AsyncModelsResource(_base._BaseResource):
    """The `models` resource of `AsyncLLMGateway`."""

    async def list(self) -> models.LLMGatewayModelList:
        """
        Lists the models available through the LLM Gateway.

        Raises: `LLMGatewayError` if the request fails.
        """
        return await async_api.list_models(
            self._gateway.client.http_client,
            **self._http_options,
        )


class AsyncCompletionsResource(_base._BaseResource):
    """The `chat.completions` resource of `AsyncLLMGateway`."""

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> models.LLMGatewayChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[models.LLMGatewayCompletionChunk]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: bool,
        **kwargs: Any,
    ) -> Union[
        models.LLMGatewayChatCompletion, AsyncIterator[models.LLMGatewayCompletionChunk]
    ]: ...

    async def create(
        self,
        *,
        model: str,
        messages: List[LLMGatewayMessageParam],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        models.LLMGatewayChatCompletion, AsyncIterator[models.LLMGatewayCompletionChunk]
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

        With `stream=True`, resolves to an async iterator of
        `LLMGatewayCompletionChunk` instead of a single
        `LLMGatewayChatCompletion`. Chunk parsing is verified for
        OpenAI-routed models only.

        Raises: `LLMGatewayError` if the request fails (for a stream, only if
            it fails before any bytes are received — a failure mid-stream
            just truncates the iterator).
        """
        body = _base.completion_body(
            model=model, messages=messages, stream=stream, extra=kwargs
        )

        if stream:
            return async_api.stream_completion(
                self._gateway.client.http_client,
                body=body,
                **self._http_options,
            )

        return await async_api.create_completion(
            self._gateway.client.http_client,
            body=body,
            **self._http_options,
        )


class AsyncChatResource:
    """The `chat` resource of `AsyncLLMGateway`."""

    def __init__(self, gateway: "AsyncLLMGateway") -> None:
        self.completions = AsyncCompletionsResource(gateway)


class AsyncUnderstandingResource(_base._BaseResource):
    """The `understanding` resource of `AsyncLLMGateway`."""

    async def create(
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
        return await async_api.create_understanding(
            self._gateway.client.http_client,
            body=_base.understanding_body(
                transcript_id=transcript_id, request=request, extra=kwargs
            ),
            **self._http_options,
        )

    async def validate(
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
        await async_api.validate_understanding(
            self._gateway.client.http_client,
            body=_base.validate_body(
                request=request, transcript_id=transcript_id, extra=kwargs
            ),
            **self._http_options,
        )


class AsyncLLMGateway:
    """
    The asyncio counterpart of `LLMGateway`.

    The gateway owns an HTTP connection pool. Close it with `aclose()`, or
    use the gateway as an async context manager.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncLLMGateway() as gateway:
                models = await gateway.models.list()
                for model in models.data:
                    print(model.id)

                completion = await gateway.chat.completions.create(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "Summarize this call."}],
                )
                print(completion.choices[0].message.content)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates an `AsyncLLMGateway`.

        Args:
            client: The `AsyncClient` to use. If `None`, the gateway creates
                one from the global `settings` and closes it on `aclose()`.
                Pass a client to share one pool between gateways.
            api_key: The API key to authenticate with. The gateway builds its
                own `AsyncClient` from it and closes that client on
                `aclose()`. Given alongside `client`, it takes precedence: the
                gateway builds and owns a client made from a copy of that
                client's settings with the key replaced, and the given client
                is left untouched and stays the caller's to close.
        """
        self._owns_client = client is None or api_key is not None
        self._client = _async_client._resolve_client(client, api_key)
        self.models = AsyncModelsResource(self)
        self.chat = AsyncChatResource(self)
        self.understanding = AsyncUnderstandingResource(self)

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this gateway sends requests with."""

        return self._client

    async def aclose(self) -> None:
        """
        Closes the HTTP connection pool.

        Leaves a client that was passed in alone. Its creator closes it.
        """
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.aclose()
