"""Backwards-compatibility tests for the sync.py → sync/v1/ package move.

The flat ``sync.py`` module became the ``sync/`` package with the
implementation in ``sync/v1/`` (mirroring ``streaming/v3/``). Every import
that worked against the old module must keep working silently:

- ``from assemblyai.sync import SyncTranscriber`` (the old module path) is
  preserved by re-exports in ``sync/__init__.py``.
- ``import assemblyai as aai; aai.SyncTranscriber`` is unchanged.
- The new canonical path is ``assemblyai.sync.v1``.
"""

import warnings

import assemblyai as aai
from assemblyai import sync as sync_pkg
from assemblyai.sync.v1 import (
    SyncTranscriber as V1SyncTranscriber,
)


def test_old_module_path_still_imports_sync_transcriber():
    """``from assemblyai.sync import SyncTranscriber`` resolves to the same class."""
    from assemblyai.sync import SyncTranscriber

    assert SyncTranscriber is V1SyncTranscriber


def test_top_level_export_is_unchanged():
    """``aai.SyncTranscriber`` is the class from the new package."""
    assert aai.SyncTranscriber is V1SyncTranscriber


def test_old_module_surface_is_preserved():
    """Every name the old flat ``sync.py`` defined is importable from ``assemblyai.sync``."""
    for name in (
        "SyncTranscriber",
        "AudioInput",
        "_PCM_SUFFIXES",
        "_resolve_audio",
        "_config_to_json",
        "_SyncTranscriberImpl",
    ):
        assert hasattr(sync_pkg, name), f"assemblyai.sync.{name} is gone"


def test_old_module_path_is_silent():
    """The compatibility re-exports must not emit deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from assemblyai.sync import (  # noqa: F401
            SyncTranscriber,
        )

        getattr(sync_pkg, "SyncTranscriber")


def test_old_sync_api_module_surface_is_preserved():
    """Every name the old flat ``sync_api.py`` defined is importable from its shim."""
    from assemblyai import sync_api
    from assemblyai.sync.v1 import api

    for name in (
        "ENDPOINT_TRANSCRIBE",
        "ENDPOINT_WARM",
        "MODEL_HEADER",
        "transcribe",
        "_error_from_response",
    ):
        assert getattr(sync_api, name) is getattr(api, name), (
            f"assemblyai.sync_api.{name} does not match sync.v1.api.{name}"
        )


def test_v1_package_exports_full_sync_surface():
    """The new ``sync.v1`` package exposes the client, alias, and sync types."""
    from assemblyai.sync import v1

    assert v1.SyncTranscriber is V1SyncTranscriber
    for name in (
        "AudioInput",
        "SyncSpeechModel",
        "SyncTranscriptError",
        "SyncTranscriptionConfig",
        "SyncTranscriptResponse",
        "SyncWord",
    ):
        assert hasattr(v1, name), f"assemblyai.sync.v1.{name} missing"
        assert name in v1.__all__
