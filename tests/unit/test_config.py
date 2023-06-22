import inspect

import pytest

import assemblyai as aai


def test_configuration_are_none_by_default():
    """
    Tests whether all configurations are None by default.
    """

    config = aai.TranscriptionConfig()
    fields = config.raw.__fields_set__ - {"language_code"}

    for name, value in inspect.getmembers(config):
        if name in fields and value is not None:
            pytest.fail(
                f"Configuration field {name} is {value} and not None by default."
            )


def test_speech_threshold_fails_if_outside_range():
    """
    Tests that an exception is raised if the value for speech_threshold is outside the range of [0, 1].
    """

    with pytest.raises(ValueError, match="speech_threshold"):
        aai.TranscriptionConfig(speech_threshold=1.5)
    with pytest.raises(ValueError, match="speech_threshold"):
        aai.TranscriptionConfig(speech_threshold=-0.5)
