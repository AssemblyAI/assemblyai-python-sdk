"""Backwards-compatibility shim for the old flat ``sync_api.py`` module.

The sync API transport layer now lives in ``sync/v1/api.py`` alongside the
rest of the sync product. This module re-exports its full surface so every
existing ``assemblyai.sync_api`` import keeps working silently.
"""

from .sync.v1.api import (  # noqa: F401
    ENDPOINT_TRANSCRIBE,
    ENDPOINT_WARM,
    MODEL_HEADER,
    _error_from_response,
    transcribe,
)

__all__ = [
    "ENDPOINT_TRANSCRIBE",
    "ENDPOINT_WARM",
    "MODEL_HEADER",
    "transcribe",
]
