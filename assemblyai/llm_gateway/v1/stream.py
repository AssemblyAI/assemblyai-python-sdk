from typing import AsyncIterator, Iterator, Optional

from typing_extensions import Self

from . import models


class LLMGatewayStreamAccumulator:
    """The message assembled from a `LLMGatewayStream`'s chunks, as seen so far."""

    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.content: str = ""
        self.finish_reason: Optional[str] = None
        self.usage: Optional[models.LLMGatewayUsage] = None

    def _absorb(self, chunk: models.LLMGatewayCompletionChunk) -> None:
        if self.id is None:
            self.id = chunk.id
        if self.model is None:
            self.model = chunk.model
        for choice in chunk.choices:
            if choice.delta.content:
                self.content += choice.delta.content
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason
        if chunk.usage is not None:
            self.usage = chunk.usage


class LLMGatewayStream:
    """
    Wraps a streamed chat completion's chunk iterator, accumulating a final
    assembled message as it's iterated.

    Iterate it exactly like the raw chunk iterator it replaces:
    `for chunk in stream: ...`. Once exhausted, call `get_final_message()` for
    the assembled text/usage/finish_reason.

    Verified for OpenAI-routed models only, like raw chunk parsing. Tool calls
    are not reconstructed here — streamed chunks don't carry `tool_calls`
    today.
    """

    def __init__(self, chunks: Iterator[models.LLMGatewayCompletionChunk]) -> None:
        self._chunks = chunks
        self._accumulator = LLMGatewayStreamAccumulator()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> models.LLMGatewayCompletionChunk:
        chunk = next(self._chunks)
        self._accumulator._absorb(chunk)
        return chunk

    def get_final_message(self) -> LLMGatewayStreamAccumulator:
        """
        Returns the message assembled from chunks seen so far.

        Call after exhausting the iterator for a complete result; calling it
        earlier just returns whatever has streamed in up to that point.
        """
        return self._accumulator


class AsyncLLMGatewayStream:
    """The asyncio counterpart of `LLMGatewayStream`."""

    def __init__(self, chunks: AsyncIterator[models.LLMGatewayCompletionChunk]) -> None:
        self._chunks = chunks
        self._accumulator = LLMGatewayStreamAccumulator()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> models.LLMGatewayCompletionChunk:
        chunk = await self._chunks.__anext__()
        self._accumulator._absorb(chunk)
        return chunk

    def get_final_message(self) -> LLMGatewayStreamAccumulator:
        """
        Returns the message assembled from chunks seen so far.

        Call after exhausting the iterator for a complete result; calling it
        earlier just returns whatever has streamed in up to that point.
        """
        return self._accumulator
