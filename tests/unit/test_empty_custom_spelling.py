import pytest

import assemblyai as aai


def test_empty_custom_spelling_constructor():
    assert aai.TranscriptionConfig(custom_spelling={}).custom_spelling is None


@pytest.mark.parametrize("override", [False, True])
def test_empty_custom_spelling_setter_when_unset(override):
    config = aai.TranscriptionConfig()
    assert config.set_custom_spelling({}, override=override) is config
    assert config.custom_spelling is None


def test_empty_custom_spelling_clears_existing():
    config = aai.TranscriptionConfig(custom_spelling={"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling({})
    assert config.custom_spelling is None


def test_empty_custom_spelling_merge_keeps_existing():
    config = aai.TranscriptionConfig(custom_spelling={"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling({}, override=False)
    assert config.custom_spelling == {"AssemblyAI": ["assemblyAI"]}


@pytest.mark.parametrize("override", [False, True])
def test_none_custom_spelling_controls(override):
    assert aai.TranscriptionConfig(custom_spelling=None).custom_spelling is None
    config = aai.TranscriptionConfig(custom_spelling={"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling(None, override=override)
    assert config.custom_spelling is None


def test_nonempty_custom_spelling_merge():
    config = aai.TranscriptionConfig(custom_spelling={"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling({"Kubernetes": ["k8s", "kubernetes"]}, override=False)
    assert config.custom_spelling == {
        "AssemblyAI": ["assemblyAI"],
        "Kubernetes": ["k8s", "kubernetes"],
    }
    config.set_custom_spelling({"AssemblyAI": ["assembly ai"]}, override=False)
    assert config.custom_spelling == {
        "AssemblyAI": ["assembly ai"],
        "Kubernetes": ["k8s", "kubernetes"],
    }
