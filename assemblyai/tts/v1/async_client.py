from typing import Optional

from ... import async_client as _async_client
from . import async_api
from ._base import check_config, check_text, request_body, synthesize_timeout
from .async_api import AsyncTTSStream
from .models import TTSConfig, TTSResponse

__all__ = ["AsyncTextToSpeech"]


class AsyncTextToSpeech:
    """
    Text in, speech out, for asyncio: the counterpart of `TextToSpeech`.

    Same options, result and errors; `synthesize()` is a coroutine and
    `stream()` returns an `AsyncTTSStream`. Owns an HTTP pool: use
    `async with` or `await aclose()`, or pass an `aai.AsyncClient` to share
    one.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncTextToSpeech() as tts:
                async with tts.stream("Hello from AssemblyAI.") as speech:
                    async for chunk in speech:
                        await play(chunk)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        config: Optional[TTSConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates an `AsyncTextToSpeech`.

        Args:
            client: The `AsyncClient` to use. If `None`, one is created from
                the global `settings` and closed on `aclose()`. Pass a client
                to share one pool between instances.
            config: Default options. A per-call `config` replaces it whole,
                so repeat the voice_id in a per-call config.
            api_key: The API key to authenticate with. Builds and owns an
                `AsyncClient` from it. Given alongside `client`, it takes
                precedence: a client is built from a copy of that client's
                settings with the key replaced, and the given client is left
                untouched and stays the caller's to close.

        Raises:
            TypeError: if `config` is not a `TTSConfig`.
        """
        check_config(type(self).__name__, config)

        self._owns_client = client is None or api_key is not None
        self._client = _async_client._resolve_client(client, api_key)
        self.config = config or TTSConfig()

    @property
    def config(self) -> TTSConfig:
        """The default configuration of the `AsyncTextToSpeech`."""
        return self._config

    @config.setter
    def config(self, config: TTSConfig) -> None:
        check_config(type(self).__name__, config)

        self._config = config or TTSConfig()

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this instance sends requests with."""

        return self._client

    async def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSResponse:
        """
        Speaks `text` and returns the complete audio.

        The asyncio counterpart of `TextToSpeech.synthesize`.

        Raises:
            TTSError: on any API refusal.
        """
        check_text(text)
        check_config(type(self).__name__, config)
        body = request_body(text, config or self._config)

        return await async_api.synthesize(
            self._client.http_client,
            base_url=self._client.settings.tts_base_url,
            body=body,
            timeout=synthesize_timeout(self._client.settings),
        )

    def stream(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> AsyncTTSStream:
        """
        Speaks `text` and returns the audio as it is generated.

        The asyncio counterpart of `TextToSpeech.stream`. Nothing is sent
        until the returned `AsyncTTSStream` is entered with `async with` or
        first read, which is where a refusal raises. Use `async with`, or
        `await aclose()` it, so the connection is released.

        Raises:
            TTSError: from the `async with` or the first read, if the API
                refuses the request before any audio.
        """
        check_text(text)
        check_config(type(self).__name__, config)
        body = request_body(text, config or self._config)

        return async_api.stream_speech(
            self._client.http_client,
            base_url=self._client.settings.tts_base_url,
            body=body,
            timeout=self._client.settings.tts_http_timeout,
        )

    async def aclose(self) -> None:
        """Closes the HTTP pool this instance owns. A shared `client` is left open."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncTextToSpeech":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
