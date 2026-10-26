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


def test_raw_transcription_config_values_are_kept():
    """
    Tests that the values of a passed `raw_transcription_config` are not
    overwritten by the `None` defaults of the other constructor arguments.
    """

    raw = aai.RawTranscriptionConfig(
        language_code="de",
        punctuate=False,
        custom_spelling=[{"from": ["k8s"], "to": "Kubernetes"}],
    )

    config = aai.TranscriptionConfig(raw_transcription_config=raw)

    assert config.raw is raw
    assert config.language_code == "de"
    assert config.punctuate is False
    assert config.custom_spelling == {"Kubernetes": ["k8s"]}


def test_explicit_arguments_override_raw_transcription_config():
    """
    Tests that explicit constructor arguments take priority over the values of
    a passed `raw_transcription_config`, and the others are kept.
    """

    raw = aai.RawTranscriptionConfig(language_code="de", punctuate=False)

    config = aai.TranscriptionConfig(raw_transcription_config=raw, language_code="fr")

    assert config.language_code == "fr"
    assert config.punctuate is False
