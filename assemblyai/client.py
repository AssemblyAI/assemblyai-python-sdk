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


_MISSING_API_KEY_ERROR = (
    "Please provide an API key: set the ASSEMBLYAI_API_KEY environment variable, "
    "set aai.settings.api_key, or pass api_key= to the transcriber or client."
)


class Client:
    _default: ClassVar[Optional["Client"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        settings: Optional[types.Settings] = None,
        api_key: Optional[str] = None,
        api_key_required: bool = True,
    ) -> None:
        """
        Creates the AssemblyAI client.

        Args:
            settings: The settings to use for the client. If `None` is given, the global
                settings are used. The client holds a copy, so the given settings object
                is never modified.
            api_key: The API key to authenticate with. Overrides the key on `settings`.
            api_key_required: If an API key is required (either as environment variable or the global settings).
                Can be set to `False` if a different authentication method is used, e.g., a temporary token.
        """
        from . import settings as default_settings

        self._settings = (settings if settings is not None else default_settings).copy()

        if api_key is not None:
            self._settings.api_key = api_key

        if api_key_required and not self._settings.api_key:
            raise ValueError(_MISSING_API_KEY_ERROR)

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


def _resolve_client(
    client: Optional[Client],
    api_key: Optional[str],
) -> Client:
    """
    Returns the client a transcriber sends its requests with.

    `api_key` takes precedence. A client's credentials are baked into its
    connection pool when it is built, so a key given alongside a `client`
    derives a new client from a copy of that client's settings rather than
    changing anything on it. The given client is left untouched and unused.

    A client returned here is the transcriber's own except when it is the
    caller's `client` passed without an `api_key`.

    Args:
        client: An explicit `Client`, or `None`.
        api_key: An API key, or `None`.
    """

    if client is not None:
        if api_key is not None:
            # `Client.__init__` copies the settings it is given.
            return Client(settings=client.settings, api_key=api_key)

        return client

    if api_key is not None:
        return Client(api_key=api_key)

    return Client.get_default()
