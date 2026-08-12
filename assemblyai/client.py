import sys
import threading
from typing import ClassVar, Dict, Optional

import httpx

from . import types
from .__version__ import __version__


def _build_headers(settings: types.Settings) -> Dict[str, str]:
    """
    Builds the headers every request carries. Shared with `AsyncClient`.
    """

    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    user_agent = f"{httpx._client.USER_AGENT} AssemblyAI/1.0 (sdk=Python/{__version__} runtime_env=Python/{python_version})"

    headers = {"user-agent": user_agent}
    if settings.api_key:
        headers["authorization"] = settings.api_key

    return headers


def _build_limits(settings: types.Settings) -> httpx.Limits:
    """Builds the pool limits from `settings.keepalive_expiry`."""

    keepalive_expiry = settings.keepalive_expiry

    return (
        httpx.Limits(keepalive_expiry=keepalive_expiry)
        if keepalive_expiry is not None
        else httpx.Limits()
    )


class Client:
    _default: ClassVar[Optional["Client"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        settings: types.Settings,
        api_key_required: bool = True,
    ) -> None:
        """
        Creates the AssemblyAI client.

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

        def _store_response(response):
            self._last_response = response

        self._http_client = httpx.Client(
            base_url=self.settings.base_url,
            headers=_build_headers(self._settings),
            timeout=self.settings.http_timeout,
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
    def http_client(self) -> httpx.Client:
        """
        Get the current HTTP client.

        Returns:
            The current HTTP client.
        """

        return self._http_client

    @classmethod
    def get_default(cls, api_key_required: bool = True):
        """
        Return the default client.

        Args:
            api_key_required: If the default client requires an API key.

        Returns:
            The default client with the default settings.
        """
        from . import settings as default_settings

        if cls._default is None or cls._default.settings != default_settings:
            with cls._lock:
                if cls._default is None or cls._default.settings != default_settings:
                    cls._default = cls(
                        settings=default_settings, api_key_required=api_key_required
                    )

        return cls._default
