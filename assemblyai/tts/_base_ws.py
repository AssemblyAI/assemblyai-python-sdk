"""Helpers shared by the sync and async Text-to-Speech sessions."""

import asyncio
import platform
import ssl
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import urlencode, urlsplit

from .. import _audio
from ..__version__ import __version__
from ..types import Settings
from .v1.api import _RETRYABLE_STATUSES
from .v1.models import TTSConfig, TTSContainer, TTSError

# The route of the WebSocket API, after the host and any path prefix of
# `settings.tts_base_url`.
WS_PATH = "/v1/ws"
# Characters a single `Generate` frame may carry (Unicode code points) when
# `Begin` does not report the limit in force.
MAX_GENERATE_CHARS = 2000
# Audio frames queued for pulling before the oldest is dropped: about 160 s
# at the API's ~80 ms per frame. Bounds memory when audio is not read, or is
# read more slowly than it arrives.
PULL_QUEUE_FRAMES = 2000

# An item `audio()` reads: (kind, flush_id or cancel id, payload). `audio`
# items carry a request's audio; the markers `done` and `cancelled` end a
# request, and `error` and `closed` end the session.
PullItem = Tuple[str, Any, Any]


class PullBuffer:
    """
    The items `audio()` reads, in arrival order.

    Markers are always kept, since an `audio()` iteration ends only at one.
    Past `max_audio` audio frames, the oldest audio frame is dropped.
    """

    def __init__(self, max_audio: int) -> None:
        self.max_audio = max_audio
        self._items: Deque[PullItem] = deque()
        self._audio = 0

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: PullItem) -> bool:
        """Adds `item`; returns whether an older audio frame was dropped."""

        self._items.append(item)
        if item[0] != "audio":
            return False
        self._audio += 1
        if self._audio <= self.max_audio:
            return False
        for index, queued in enumerate(self._items):
            if queued[0] == "audio":
                del self._items[index]
                self._audio -= 1
                return True
        return False

    def popleft(self) -> Optional[PullItem]:
        """Removes and returns the oldest item; `None` when there is none."""

        if not self._items:
            return None
        item = self._items.popleft()
        if item[0] == "audio":
            self._audio -= 1
        return item


def max_generate_chars(begin: Any) -> int:
    """
    The characters one `Generate` may carry: the limit the session's
    `Begin` reports (`limits.max_generate_text_length`, where `0` means
    none), else `MAX_GENERATE_CHARS`.
    """

    configuration = getattr(begin, "configuration", None) or {}
    limits = configuration.get("limits")
    limit = limits.get("max_generate_text_length") if isinstance(limits, dict) else None
    if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
        return limit
    return MAX_GENERATE_CHARS


def check_text_frame(text: object, limit: int = MAX_GENERATE_CHARS) -> None:
    """Refuses what one `Generate` cannot carry; a `limit` of `0` is none."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    if limit and len(text) > limit:
        raise ValueError(
            f"text is {len(text)} characters, over the {limit}-character "
            "limit of one Generate frame; split it across several generate() calls"
        )


def resolve_settings(
    owner: str, base: Optional[Settings], api_key: Optional[str], token: Optional[str]
) -> Settings:
    """
    The settings a session connects with: a copy, so nothing shared changes.

    `base` is a client's settings, else the global `assemblyai.settings`;
    `api_key` replaces the copy's key. Without a key or a `token`, raises
    the SDK's usual message. No HTTP client is built: a session needs none.
    """

    if base is None:
        from .. import settings as _global

        base = _global
    settings = base.model_copy() if hasattr(base, "model_copy") else base.copy()
    if api_key is not None:
        settings.api_key = api_key
    if token is None and not settings.api_key:
        raise ValueError(
            "Please provide an API key: set the ASSEMBLYAI_API_KEY environment "
            f"variable, set aai.settings.api_key, or pass api_key= or token= to {owner}."
        )
    return settings


def handshake_error(exc: BaseException) -> TTSError:
    """
    The `TTSError` for a WebSocket handshake that failed with `exc`.

    An HTTP answer is retryable by its status, as on the HTTP API; with no
    answer, a refused or dropped connection and a timeout are retryable.
    """

    # The legacy client's InvalidStatusCode carries `status_code`; the
    # current clients' InvalidStatus carries the `response`.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        retryable = status in _RETRYABLE_STATUSES
    else:
        from websockets.exceptions import InvalidMessage  # a dropped handshake

        status = None
        retryable = isinstance(
            exc, (OSError, asyncio.TimeoutError, InvalidMessage)
        ) and not isinstance(exc, ssl.CertificateError)
    return TTSError(
        f"WebSocket handshake failed: {exc}", status_code=status, retryable=retryable
    )


def _user_agent() -> str:
    return (
        f"AssemblyAI/1.0 (sdk=Python/{__version__} "
        f"runtime_env=Python/{platform.python_version()})"
    )


def check_session_config(config: TTSConfig) -> None:
    """Refuses the options a session cannot honour: it streams raw audio."""

    if config.container not in (None, TTSContainer.raw) or config.bit_rate is not None:
        raise ValueError(
            "a wav container and bit_rate do not apply to a session, which streams "
            "raw pcm_s16le, pcm_mulaw or pcm_alaw; leave them unset in its TTSConfig"
        )


def build_uri(
    base_url: str,
    config: TTSConfig,
    *,
    token: Optional[str],
    api_key: Optional[str],
    word_boundaries: bool,
    inactivity_timeout: Optional[int],
) -> Tuple[str, Dict[str, Optional[str]]]:
    """The session URI and handshake headers from the settings and config."""

    check_session_config(config)
    parts = urlsplit(base_url if "://" in base_url else f"https://{base_url}")
    scheme = "ws" if parts.scheme in ("http", "ws") else "wss"
    params: Dict[str, Any] = (
        _audio._config_to_json(config, exclude=("container", "bit_rate")) or {}
    )
    params["word_boundaries"] = word_boundaries
    if inactivity_timeout is not None:
        params["inactivity_timeout"] = inactivity_timeout
    if token is not None:
        params["token"] = token
    # The API reads booleans case-insensitively; send the canonical form.
    query = urlencode(
        {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in params.items()}
    )
    headers: Dict[str, Optional[str]] = {"User-Agent": _user_agent()}
    if token is None and api_key:
        headers["Authorization"] = api_key
    prefix = parts.path.rstrip("/")
    return f"{scheme}://{parts.netloc}{prefix}{WS_PATH}?{query}", headers
