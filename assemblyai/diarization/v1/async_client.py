import asyncio
import functools
from typing import Optional

from ... import async_client as _async_client
from . import async_api
from ._base import AudioInput, TranscriptInput, check_config, resolve_request
from .models import DiarizationConfig, DiarizationResponse

__all__ = ["AsyncDiarizer"]


class AsyncDiarizer:
    """
    Speaker labels for audio, for asyncio: the counterpart of `Diarizer`.

    Same options, result and errors; `diarize()` is a coroutine, and a path
    or file object is read on a worker thread. Owns an HTTP pool: use
    `async with` or `await aclose()`, or pass an `aai.AsyncClient` to share
    one.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncDiarizer() as diarizer:
                result = await diarizer.diarize("call.wav")
                for segment in result.segments:
                    print(segment.speaker, segment.start, segment.end)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        config: Optional[DiarizationConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates an `AsyncDiarizer`.

        Args:
            client: The `AsyncClient` to use. If `None`, one is created from
                the global `settings` and closed on `aclose()`. Pass a client
                to share one pool between instances.
            config: Default options. A per-call `config` replaces it whole.
            api_key: The API key to authenticate with. Builds and owns an
                `AsyncClient` from it. Given alongside `client`, it takes
                precedence and the given client is left untouched.

        Raises:
            TypeError: if `config` is not a `DiarizationConfig`.
        """
        check_config(type(self).__name__, config)

        self._owns_client = client is None or api_key is not None
        self._client = _async_client._resolve_client(client, api_key)
        self.config = config or DiarizationConfig()

    @property
    def config(self) -> DiarizationConfig:
        """The default configuration of the `AsyncDiarizer`."""
        return self._config

    @config.setter
    def config(self, config: DiarizationConfig) -> None:
        check_config(type(self).__name__, config)

        self._config = config or DiarizationConfig()

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this instance sends requests with."""

        return self._client

    async def diarize(
        self,
        audio: AudioInput,
        transcript: Optional[TranscriptInput] = None,
        config: Optional[DiarizationConfig] = None,
        *,
        content_type: Optional[str] = None,
    ) -> DiarizationResponse:
        """
        Labels the speakers in `audio`.

        The asyncio counterpart of `Diarizer.diarize`; same arguments.

        Raises:
            DiarizationError: on any API refusal.
        """
        (
            data,
            filename,
            audio_content_type,
            transcript_json,
            config_json,
        ) = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                resolve_request,
                type(self).__name__,
                audio,
                transcript,
                config or self._config,
                content_type,
            ),
        )

        return await async_api.diarize(
            self._client.http_client,
            base_url=self._client.settings.diarization_base_url,
            audio=data,
            filename=filename,
            audio_content_type=audio_content_type,
            transcript=transcript_json,
            config=config_json,
            timeout=self._client.settings.diarization_http_timeout,
        )

    async def aclose(self) -> None:
        """Closes the HTTP pool this instance owns. A shared `client` is left open."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncDiarizer":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
