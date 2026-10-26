import os
import ssl
import sys
import threading
import weakref
from typing import Any, ClassVar, Dict, Optional

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


def _build_tts_limits(settings: types.Settings) -> httpx.Limits:
    """
    Builds the limits of the Text-to-Speech pool: no cap on connections, as
    from `_build_limits`, and idle connections kept
    `settings.tts_keepalive_expiry` seconds. Shared with `AsyncClient`.
    """

    return httpx.Limits(keepalive_expiry=settings.tts_keepalive_expiry)


# What httpx raises from a request on a closed client, raised as well when a
# client closed before it built its Text-to-Speech pool is asked for that pool.
_CLOSED_CLIENT_ERROR = "Cannot send a request, as the client has been closed."


# The context `_shared_tts_ssl_context()` returns once it has been built.
_tts_ssl_context: Optional[ssl.SSLContext] = None
_TTS_SSL_CONTEXT_LOCK = threading.Lock()


def _shared_tts_ssl_context() -> ssl.SSLContext:
    """
    Returns the TLS context of every Text-to-Speech pool in the process,
    built on the first call. Shared with `AsyncClient`.

    It is the context httpx would build for the pool by default (certifi's CA
    bundle, or `SSL_CERT_FILE` / `SSL_CERT_DIR` when set), built once instead
    of once per pool. A build loads and parses the CA bundle, about 15 ms on an
    idle machine with OpenSSL 3.0 and several times that on a busy one, and
    each `AsyncTextToSpeech` built without `client=`, like each `TextToSpeech`
    built from `api_key=`, has a pool of its own. One context serves any
    number of pools, threads and event loops; each connection keeps its own
    TLS state.

    Only HTTP/1.1 pools may use it: httpcore sets the context's ALPN protocols
    at every connect, to `http/1.1` for an HTTP/1.1 pool.

    The CA certificates and environment it reflects are those in place at the
    first call; a later change takes effect in a new process.
    """

    global _tts_ssl_context
    context = _tts_ssl_context
    if context is None:
        with _TTS_SSL_CONTEXT_LOCK:
            context = _tts_ssl_context
            if context is None:
                context = _tts_ssl_context = httpx.create_ssl_context()
    return context


# Every `Client` and `AsyncClient` that has built a Text-to-Speech pool, for
# `_forget_tts_pools_in_child()`.
_TTS_POOL_HOLDERS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _forget_tts_pools_in_child() -> None:
    """
    Runs in a child process right after `fork()`, and drops every
    Text-to-Speech pool the child inherited, so the child's next request
    builds a pool of its own.

    An inherited pool holds the parent's idle connections, kept up to
    `tts_keepalive_expiry` seconds (120 by default): a request made before
    the fork, such as a warm-up in a `gunicorn --preload` app, leaves one
    there. Processes sending on one connection at once garble each other's
    responses. The pools are dropped rather than closed: another thread of the
    parent may have held a pool's lock at the moment of the fork, and an async
    pool needs its event loop. Each process closes its own socket descriptors,
    so the parent's connections stay intact.

    Dropping relies on httpx not closing a client when it is garbage
    collected, which holds from httpx 0.22, the floor in `setup.py`. httpx
    0.19 to 0.21 close it in `Client.__del__`, which takes the pool's lock.
    Keeping the dropped pools referenced would not help there: the child's
    interpreter would close them at exit, and its copies of the parent's
    sockets would stay open until then.
    """

    for holder in list(_TTS_POOL_HOLDERS):
        holder._forget_tts_http_client()
    _TTS_POOL_HOLDERS.clear()


if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_forget_tts_pools_in_child)


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

        self._event_hooks = {"response": [_store_response]}
        self._http_client = httpx.Client(
            base_url=self.settings.base_url,
            headers=_build_headers(self._settings),
            timeout=self.settings.http_timeout,
            limits=_build_limits(self._settings),
            event_hooks=self._event_hooks,
        )

        # Built when a `TextToSpeech` takes this client, or on first use, so
        # a client that never speaks pays nothing for it.
        self._tts_http_client: Optional[httpx.Client] = None
        self._tts_http_client_closed = False
        self._tts_http_client_lock = threading.Lock()

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

    @property
    def tts_http_client(self) -> httpx.Client:
        """
        The HTTP client of the Text-to-Speech API, built when a
        `TextToSpeech` takes this client or on first use.

        A connection pool of its own, whose idle connections stay
        `settings.tts_keepalive_expiry` seconds (120 by default): a voice
        agent's next sentence reuses the connection of the previous one
        instead of opening a new one. `http_client` and its
        `settings.keepalive_expiry` are unaffected. Responses update
        `last_response` as those of `http_client` do. `close()` closes it
        with `http_client`.

        Raises:
            RuntimeError: if the pool was closed before it was ever built.
        """

        client = self._ensure_tts_http_client()
        if client is None:
            raise RuntimeError(_CLOSED_CLIENT_ERROR)

        return client

    def _ensure_tts_http_client(self) -> Optional[httpx.Client]:
        """
        Returns the Text-to-Speech pool, building it if it is not built yet,
        or `None` if the pool was closed before it was ever built.

        `TextToSpeech` calls it when it is constructed, so its first request
        does not wait for the build.
        """

        client = self._tts_http_client
        if client is None:
            with self._tts_http_client_lock:
                client = self._tts_http_client
                if client is None and not self._tts_http_client_closed:
                    client = self._tts_http_client = httpx.Client(
                        base_url=self._settings.tts_base_url,
                        headers=_build_headers(self._settings),
                        timeout=self._settings.tts_http_timeout,
                        limits=_build_tts_limits(self._settings),
                        event_hooks=self._event_hooks,
                        verify=_shared_tts_ssl_context(),
                    )
                    _TTS_POOL_HOLDERS.add(self)

        return client

    def _forget_tts_http_client(self) -> None:
        """Drops the Text-to-Speech pool unclosed; see `_forget_tts_pools_in_child`."""

        self._tts_http_client = None
        # Another thread of the parent may have held the lock at the fork.
        self._tts_http_client_lock = threading.Lock()

    def _close_tts_http_client(self) -> None:
        """Closes the Text-to-Speech pool, and keeps it from being built later."""

        with self._tts_http_client_lock:
            self._tts_http_client_closed = True
            client = self._tts_http_client
        if client is not None:
            client.close()

    def close(self) -> None:
        """
        Closes both HTTP connection pools, `http_client` and `tts_http_client`.

        A request after it raises httpx's closed-client `RuntimeError`. Leave
        the default client (`Client.get_default()`) open: every instance built
        without `client=` or `api_key=` shares it.
        """

        try:
            self._http_client.close()
        finally:
            self._close_tts_http_client()

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
