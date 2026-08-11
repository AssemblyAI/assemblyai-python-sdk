"""Backwards-compatibility tests for the prerecorded transcription surface.

``assemblyai.transcriber`` re-exports the prerecorded surface whose canonical
location is ``assemblyai.prerecorded.v2``. Every import that names the flat
module must keep working silently:

- ``from assemblyai.transcriber import Transcriber`` (the flat module path) is
  preserved by re-exports in ``transcriber.py``.
- ``import assemblyai as aai; aai.Transcriber`` is unchanged.
- The canonical path is ``assemblyai.prerecorded.v2``.
"""

import warnings

import assemblyai as aai
from assemblyai import transcriber as transcriber_module
from assemblyai.prerecorded import v2
from assemblyai.prerecorded.v2 import client as v2_client
from assemblyai.prerecorded.v2 import transcript as v2_transcript
from assemblyai.prerecorded.v2 import transcript_group as v2_transcript_group


def test_old_module_path_still_imports_prerecorded_classes():
    """``from assemblyai.transcriber import ...`` resolves to the same classes."""
    from assemblyai.transcriber import Transcriber, Transcript, TranscriptGroup

    assert Transcriber is v2.Transcriber is v2_client.Transcriber
    assert Transcript is v2.Transcript is v2_transcript.Transcript
    assert TranscriptGroup is v2.TranscriptGroup is v2_transcript_group.TranscriptGroup


def test_top_level_exports_are_unchanged():
    """``aai.Transcriber`` and friends are the classes from the new package."""
    assert aai.Transcriber is v2.Transcriber
    assert aai.Transcript is v2.Transcript
    assert aai.TranscriptGroup is v2.TranscriptGroup


def test_old_module_surface_is_preserved():
    """Every name the flat ``transcriber.py`` exposes matches its canonical module."""
    for name, module in (
        ("Transcriber", v2_client),
        ("Transcript", v2_transcript),
        ("TranscriptGroup", v2_transcript_group),
        ("_TranscriberImpl", v2_client),
        ("_TranscriptGroupImpl", v2_transcript_group),
        ("_TranscriptImpl", v2_transcript),
    ):
        assert hasattr(transcriber_module, name), (
            f"assemblyai.transcriber.{name} is gone"
        )
        assert getattr(transcriber_module, name) is getattr(module, name), (
            f"assemblyai.transcriber.{name} does not match {module.__name__}.{name}"
        )


def test_old_module_path_is_silent():
    """The compatibility re-exports must not emit deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from assemblyai.prerecorded.v2 import (  # noqa: F401
            Transcriber as V2Transcriber,
        )
        from assemblyai.transcriber import (  # noqa: F401
            Transcriber,
        )

        getattr(transcriber_module, "Transcriber")


def test_root_api_module_reexports_prerecorded_endpoints():
    """Every prerecorded endpoint is importable from ``assemblyai.api`` unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from assemblyai import api as root_api
        from assemblyai.prerecorded.v2 import api as v2_api

    for name in (
        "ENDPOINT_TRANSCRIPT",
        "create_transcript",
        "delete_transcript",
        "export_subtitles_srt",
        "export_subtitles_vtt",
        "get_paragraphs",
        "get_redacted_audio",
        "get_sentences",
        "get_transcript",
        "list_transcripts",
        "word_search",
    ):
        assert getattr(root_api, name) is getattr(v2_api, name), (
            f"assemblyai.api.{name} does not match prerecorded.v2.api.{name}"
        )

    for name in ("ENDPOINT_UPLOAD", "upload_file", "lemur_task", "_get_error_message"):
        assert hasattr(root_api, name), f"assemblyai.api.{name} is gone"


def test_v2_package_exports_full_prerecorded_surface():
    """The ``prerecorded.v2`` package exposes the client, result types, and config."""
    assert v2.Transcriber is v2_client.Transcriber
    for name in (
        "Transcriber",
        "Transcript",
        "TranscriptGroup",
        "TranscriptionConfig",
    ):
        assert hasattr(v2, name), f"assemblyai.prerecorded.v2.{name} missing"
        assert name in v2.__all__
