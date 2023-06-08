import threading
from typing import ClassVar, Optional

import httpx
from typing_extensions import Self

from . import types


class Client:
    _default: ClassVar[Optional["Client"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        settings: types.Settings,
    ) -> None:
        """
        Creates the AssemblyAI client.

        Args:
            settings: The settings to use for the client.
        """

        self._settings = settings.copy()

        if not self._settings.api_key:
            raise ValueError(
                "Please provide an API key via the ASSEMBLYAI_API_KEY environment variable or the global settings."
            )

        self._http_client = httpx.Client(
            base_url=self.settings.base_url,
            headers={
                "authorization": self.settings.api_key,
            },
            timeout=self.settings.http_timeout,
        )

    @property
    def settings(self) -> types.Settings:
        """
        Get the current settings.

        Returns:
            The current settings.
        """

        return self._settings

    @property
    def http_client(self) -> httpx.Client:
        """
        Get the current HTTP client.

        Returns:
            The current HTTP client.
        """

        return self._http_client

    @classmethod
    def get_default(cls) -> Self:
        """
        Return the default client.

        Returns:
            The default client with the default settings
        """
        from . import settings as default_settings

        if cls._default is None or cls._default.settings != default_settings:
            with cls._lock:
                if cls._default is None or cls._default.settings != default_settings:
                    cls._default = cls(settings=default_settings)

        return cls._default
