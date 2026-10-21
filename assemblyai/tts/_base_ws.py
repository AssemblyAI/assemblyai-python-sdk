"""Helpers shared by the sync and async Text-to-Speech sessions."""

import platform
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlsplit

from .. import _audio
from ..__version__ import __version__
from ..types import Settings
from .v1.models import TTSConfig

# The route of the WebSocket API; the host comes from `settings.tts_base_url`.
WS_PATH = "/v1/ws"
# Characters a single `Generate` frame may carry (Unicode code points).
MAX_GENERATE_CHARS = 2000
# Audio frames queued for pulling before the oldest is dropped: about 160 s
# at the API's ~80 ms per frame. Bounds memory when nothing reads them.
PULL_QUEUE_FRAMES = 2000


def check_text_frame(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    if len(text) > MAX_GENERATE_CHARS:
        raise ValueError(
            f"text is {len(text)} characters, over the {MAX_GENERATE_CHARS}-character "
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


def _user_agent() -> str:
    return (
        f"AssemblyAI/1.0 (sdk=Python/{__version__} "
        f"runtime_env=Python/{platform.python_version()})"
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
    return f"{scheme}://{parts.netloc}{WS_PATH}?{query}", headers
