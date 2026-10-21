from __future__ import annotations

from typing import Any, Dict, Optional

from ... import _audio
from .models import TTSConfig

__all__ = ["check_config", "request_body"]


def check_config(owner: str, config: Optional[TTSConfig]) -> None:
    """
    Raises unless `config` is a `TTSConfig` or `None`.

    Args:
        owner: the class to name in the message, e.g. `TextToSpeech`.
        config: the configuration to check.
    """

    _audio.check_config(owner, config, TTSConfig)


def request_body(text: str, config: TTSConfig) -> Dict[str, Any]:
    """The JSON body of a speech request: `text` plus every option that is set."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")

    body: Dict[str, Any] = _audio._config_to_json(config) or {}
    body["text"] = text
    return body
