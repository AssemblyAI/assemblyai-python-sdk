"""Tests for the audio helpers the sync and Dictation APIs share.

``assemblyai/_audio.py`` holds the input reading, Content-Type selection, and
config serialization both single-request products need. Each product package
wraps it with its own config type and error wording, so the wrappers must keep
the exact names, signatures, and wire behavior their callers rely on.
"""

import assemblyai as aai
from assemblyai import _audio
from assemblyai.dictation.v1 import _base as dictation_base
from assemblyai.sync.v1 import _base as sync_base


def test_sync_base_still_exposes_the_helper_names():
    """Every name ``sync/v1/_base.py`` published is still importable from it."""
    for name in (
        "AudioInput",
        "_PCM_SUFFIXES",
        "_resolve_audio",
        "_config_to_json",
        "check_config",
        "_SyncTranscriberImpl",
    ):
        assert hasattr(sync_base, name), f"assemblyai.sync.v1._base.{name} is gone"


def test_sync_audio_input_is_the_shared_alias():
    # Given the shared alias, Then sync re-exports the same object
    assert sync_base.AudioInput is _audio.AudioInput


def test_sync_resolve_audio_defaults_unknown_extensions_to_wav(tmp_path):
    # Given an MP3 file and a default sync config
    audio_file = tmp_path / "call.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When resolving it for the sync API
    audio, filename, content_type = sync_base._resolve_audio(
        str(audio_file), aai.SyncTranscriptionConfig()
    )

    # Then the sync API still posts every container as WAV
    assert audio == b"ID3fake-mp3-bytes"
    assert filename == "call.mp3"
    assert content_type == "audio/wav"


def test_dictation_resolve_source_maps_the_extension(tmp_path):
    # Given the same MP3 file and a default dictation config
    audio_file = tmp_path / "call.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When resolving it for the Dictation API
    chunks, filename, content_type = dictation_base.resolve_source(
        str(audio_file), aai.DictationConfig(), "DictationTranscriber"
    )

    # Then the file is read whole as one chunk and the extension picks the
    # container Content-Type
    assert list(chunks) == [b"ID3fake-mp3-bytes"]
    assert filename == "call.mp3"
    assert content_type == "audio/mpeg"


def test_resolve_audio_selects_pcm_for_both_products():
    # Given raw bytes with sample_rate + channels
    sync_config = aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
    dictation_config = aai.DictationConfig(sample_rate=16000, channels=1)

    # When resolving for either product
    _, sync_name, sync_type = sync_base._resolve_audio(b"\x00\x01", sync_config)
    _, dictation_name, dictation_type = dictation_base.resolve_source(
        b"\x00\x01", dictation_config, "DictationTranscriber"
    )

    # Then both route the audio as raw PCM under the same default filename
    assert (sync_name, sync_type) == ("audio.pcm", "audio/pcm")
    assert (dictation_name, dictation_type) == ("audio.pcm", "audio/pcm")


def test_sync_config_to_json_still_drops_the_routing_model():
    # Given a sync config with a prompt
    config = aai.SyncTranscriptionConfig(prompt="Transcribe verbatim.")

    # When serializing the config part
    data = sync_base._config_to_json(config)

    # Then the routing model stays out of the body
    assert data == {"prompt": "Transcribe verbatim."}


def test_dictation_config_to_json_keeps_every_set_field():
    # Given a dictation config with two fields set
    config = aai.DictationConfig(language_codes=["en"], llm_instruction="Summarize.")

    # When serializing the config part
    data = dictation_base._config_to_json(config)

    # Then both survive and the unset fields are dropped
    assert data == {"language_codes": ["en"], "llm_instruction": "Summarize."}


def test_config_to_json_returns_none_when_nothing_is_set():
    # Given empty configs
    # When serializing, Then there is no config part to send
    assert sync_base._config_to_json(aai.SyncTranscriptionConfig()) is None
    assert dictation_base._config_to_json(aai.DictationConfig()) is None


def test_check_config_names_the_expected_type():
    # Given a config of the wrong product
    # When checking it, Then the message names the owner and expected type
    try:
        sync_base.check_config("SyncTranscriber", aai.DictationConfig())
    except TypeError as error:
        assert "SyncTranscriber expects SyncTranscriptionConfig" in str(error)
        assert "Use aai.SyncTranscriptionConfig." in str(error)
    else:
        raise AssertionError("check_config accepted the wrong config type")

    try:
        dictation_base.check_config(
            "DictationTranscriber", aai.SyncTranscriptionConfig()
        )
    except TypeError as error:
        assert "DictationTranscriber expects DictationConfig" in str(error)
    else:
        raise AssertionError("check_config accepted the wrong config type")


def test_check_config_accepts_none():
    # Given no config at all, Then neither product objects
    assert sync_base.check_config("SyncTranscriber", None) is None
    assert dictation_base.check_config("DictationTranscriber", None) is None


def test_resolve_source_names_the_calling_client_in_url_errors():
    # Given each dictation client's own class name
    for owner in ("DictationTranscriber", "AsyncDictationTranscriber"):
        # When a URL is passed, Then the error names that client, not a fixed one
        try:
            dictation_base.resolve_source(
                "https://example.com/audio.wav", aai.DictationConfig(), owner
            )
        except ValueError as error:
            assert str(error).startswith(f"{owner} does not accept URLs.")
        else:
            raise AssertionError("resolve_source accepted a URL")


def test_resolve_format_serves_the_streamed_path_without_audio_bytes():
    # Given only a suffix and the PCM fields, no audio
    filename, content_type = _audio.resolve_format(
        suffix="",
        filename=None,
        sample_rate=16000,
        channels=1,
        config_name="DictationConfig",
        content_types={},
    )

    # Then the part is named and typed as PCM without reading anything
    assert (filename, content_type) == ("audio.pcm", "audio/pcm")

    # And an unknown suffix falls back to WAV under the given name
    assert _audio.resolve_format(
        suffix=".xyz",
        filename="clip.xyz",
        sample_rate=None,
        channels=None,
        config_name="DictationConfig",
        content_types={},
    ) == ("clip.xyz", "audio/wav")
