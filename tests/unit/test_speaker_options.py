import pytest

import assemblyai as aai


def test_speaker_options_creation():
    """Test that SpeakerOptions can be created with valid parameters."""
    speaker_options = aai.SpeakerOptions(
        min_speakers_expected=2, max_speakers_expected=5
    )
    assert speaker_options.min_speakers_expected == 2
    assert speaker_options.max_speakers_expected == 5


def test_speaker_options_validation():
    """Test that SpeakerOptions validates max >= min."""
    with pytest.raises(
        ValueError,
        match="max_speakers_expected must be greater than or equal to min_speakers_expected",
    ):
        aai.SpeakerOptions(min_speakers_expected=5, max_speakers_expected=2)


def test_speaker_options_min_only():
    """Test that SpeakerOptions can be created with only min_speakers_expected."""
    speaker_options = aai.SpeakerOptions(min_speakers_expected=3)
    assert speaker_options.min_speakers_expected == 3
    assert speaker_options.max_speakers_expected is None


def test_speaker_options_max_only():
    """Test that SpeakerOptions can be created with only max_speakers_expected."""
    speaker_options = aai.SpeakerOptions(max_speakers_expected=5)
    assert speaker_options.min_speakers_expected is None
    assert speaker_options.max_speakers_expected == 5


def test_transcription_config_with_speaker_options():
    """Test that TranscriptionConfig accepts speaker_options parameter."""
    speaker_options = aai.SpeakerOptions(
        min_speakers_expected=2, max_speakers_expected=4
    )

    config = aai.TranscriptionConfig(
        speaker_labels=True, speaker_options=speaker_options
    )

    assert config.speaker_labels is True
    assert config.speaker_options == speaker_options
    assert config.speaker_options.min_speakers_expected == 2
    assert config.speaker_options.max_speakers_expected == 4


def test_set_speaker_diarization_with_speaker_options():
    """Test setting speaker diarization with speaker_options."""
    speaker_options = aai.SpeakerOptions(
        min_speakers_expected=1, max_speakers_expected=3
    )

    config = aai.TranscriptionConfig()
    config.set_speaker_diarization(
        enable=True, speakers_expected=2, speaker_options=speaker_options
    )

    assert config.speaker_labels is True
    assert config.speakers_expected == 2
    assert config.speaker_options == speaker_options


def test_set_speaker_diarization_disable_clears_speaker_options():
    """Test that disabling speaker diarization clears speaker_options."""
    speaker_options = aai.SpeakerOptions(min_speakers_expected=2)

    config = aai.TranscriptionConfig()
    config.set_speaker_diarization(enable=True, speaker_options=speaker_options)

    # Verify it was set
    assert config.speaker_options == speaker_options

    # Now disable
    config.set_speaker_diarization(enable=False)

    assert config.speaker_labels is None
    assert config.speakers_expected is None
    assert config.speaker_options is None


def test_speaker_options_in_raw_config():
    """Test that speaker_options is properly set in the raw config."""
    speaker_options = aai.SpeakerOptions(
        min_speakers_expected=2, max_speakers_expected=5
    )

    config = aai.TranscriptionConfig(speaker_options=speaker_options)

    assert config.raw.speaker_options == speaker_options
