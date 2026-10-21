from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ... import _audio
from ...types import Settings
from .models import TTSConfig

__all__ = ["check_config", "check_text", "request_body", "synthesize_timeout"]


def synthesize_timeout(settings: Settings) -> httpx.Timeout:
    """
    The httpx timeout of a buffered request (`synthesize()`).

    Its one read waits for the whole audio, so the read timeout is
    `tts_synthesize_http_timeout`; connecting and sending use
    `tts_http_timeout`, as a streamed request does.
    """

    return httpx.Timeout(
        settings.tts_http_timeout, read=settings.tts_synthesize_http_timeout
    )


def check_config(owner: str, config: Optional[TTSConfig]) -> None:
    """
    Raises unless `config` is a `TTSConfig` or `None`.

    Args:
        owner: the class to name in the message, e.g. `TextToSpeech`.
        config: the configuration to check.
    """

    _audio.check_config(owner, config, TTSConfig)


def check_text(text: object) -> None:
    """Raises unless `text` is a `str`; names the swapped-argument mistake."""

    if isinstance(text, TTSConfig):
        raise TypeError(
            "text comes first: synthesize(text, config) / stream(text, config)"
        )
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")


def request_body(text: str, config: TTSConfig) -> Dict[str, Any]:
    """The JSON body of a speech request: `text` plus every option that is set."""

    body: Dict[str, Any] = _audio._config_to_json(config) or {}
    body["text"] = text
    return body
