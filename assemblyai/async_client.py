from types import TracebackType
from typing import Optional, Type

import httpx
from typing_extensions import Self

from . import types
from .client import _build_headers, _build_limits


class AsyncClient:
    """
    The asyncio counterpart of `Client`. Holds an `httpx.AsyncClient`.

    `AsyncClient` has no process-wide default instance. An `httpx.AsyncClient`
    pool belongs to the event loop that first used it, so a global pool fails on
    a second event loop. `AsyncTranscriber` creates one client per instance.
    Pass an `AsyncClient` to share one pool between transcribers.

    Close the pool with `async with` or `aclose()`.

    Example:
        ```python
        import assemblyai as aai

        async with aai.AsyncClient(settings=aai.settings) as client:
            transcriber = aai.AsyncTranscriber(client=client)
            transcript = await transcriber.transcribe("./audio.mp3")
        ```
    """

    def __init__(
        self,
        *,
        settings: types.Settings,
        api_key_required: bool = True,
    ) -> None:
        """
        Creates the asyncio AssemblyAI client.

        Args:
            settings: The settings to use for the client.
            api_key_required: If an API key is required (either as environment variable or the global settings).
                Can be set to `False` if a different authentication method is used, e.g., a temporary token.
        """

        self._settings = settings.copy()

        if api_key_required and not self._settings.api_key:
            raise ValueError(
                "Please provide an API key via the ASSEMBLYAI_API_KEY environment variable or the global settings."
            )

        self._last_response: Optional[httpx.Response] = None

        async def _store_response(response: httpx.Response) -> None:
            self._last_response = response

        self._http_client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers=_build_headers(self._settings),
            timeout=self._settings.http_timeout,
            limits=_build_limits(self._settings),
            event_hooks={"response": [_store_response]},
        )

    @property
    def last_response(self) -> Optional[httpx.Response]:
        """
        Get the last HTTP response, corresponding to the last request sent from this client.

        Returns:
            The last HTTP response.
        """
        return self._last_response

    @property
    def settings(self) -> types.Settings:
        """
        Get the current settings.

        Returns:
            The current settings.
        """

        return self._settings

    @property
    def http_client(self) -> httpx.AsyncClient:
        """
        Get the current HTTP client.

        Returns:
            The current HTTP client.
        """

        return self._http_client

    async def aclose(self) -> None:
        """Closes the underlying HTTP connection pool."""

        await self._http_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.aclose()
