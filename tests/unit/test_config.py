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
