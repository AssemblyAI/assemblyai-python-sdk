import pytest

import assemblyai as aai

# `RawTranscriptionConfig` supports both pydantic v1 and v2 (assemblyai/types.py:17-34), which raise
# distinct `ValidationError` classes; both subclass `ValueError`, so assert on that to stay
# correct under either branch.


def test_speech_understanding_dict_passthrough():
    """
    Assigning a plain dict to `speech_understanding` now coerces into the typed
    `SpeechUnderstandingRequest` model, matching `SyncTranscriptionConfig`/`RealTimeParameters`.
    """
    config = aai.TranscriptionConfig()
    config.speech_understanding = {
        "request": {
            "speaker_identification": {
                "speaker_type": "name",
                "known_values": ["Michel Martin", "Peter DeCarlo"],
            }
        }
    }

    assert isinstance(config.speech_understanding, aai.SpeechUnderstandingRequest)
    identification = config.speech_understanding.request.speaker_identification
    assert identification.speaker_type == aai.SpeakerType.name
    assert identification.known_values == ["Michel Martin", "Peter DeCarlo"]


def test_language_detection_options_dict_passthrough():
    """
    Assigning a plain dict to `language_detection_options` now coerces into the typed
    `LanguageDetectionOptions` model.
    """
    config = aai.TranscriptionConfig()
    config.language_detection_options = {
        "expected_languages": ["en", "es"],
        "fallback_language": "en",
    }

    assert isinstance(config.language_detection_options, aai.LanguageDetectionOptions)
    assert config.language_detection_options.expected_languages == ["en", "es"]
    assert config.language_detection_options.fallback_language == "en"


def test_keyterms_prompt_options_dict_passthrough():
    """
    Assigning a plain dict to `keyterms_prompt_options` now coerces into the typed
    `KeytermsPromptOptions` model, consistent with the other nested-model fields.
    """
    config = aai.TranscriptionConfig()
    config.keyterms_prompt_options = {"keyterms_match_strength": "high"}

    assert isinstance(config.keyterms_prompt_options, aai.KeytermsPromptOptions)
    assert config.keyterms_prompt_options.keyterms_match_strength == "high"


def test_speaker_options_dict_passthrough_via_set_speaker_diarization():
    """
    `speaker_options` has no direct property setter -- it's only settable via
    `set_speaker_diarization()` -- so the dict-passthrough widening lives on that
    method's parameter instead.
    """
    config = aai.TranscriptionConfig().set_speaker_diarization(
        speaker_options={"min_speakers_expected": 2, "max_speakers_expected": 4}
    )

    assert isinstance(config.speaker_options, aai.SpeakerOptions)
    assert config.speaker_options.min_speakers_expected == 2
    assert config.speaker_options.max_speakers_expected == 4


def test_redact_pii_audio_options_dict_passthrough_via_set_redact_pii():
    """
    Same shape as `speaker_options`: `redact_pii_audio_options` is only settable via
    `set_redact_pii()`, not a direct property setter.
    """
    config = aai.TranscriptionConfig().set_redact_pii(
        policies=[aai.PIIRedactionPolicy.email_address],
        redact_audio_options={"override_audio_redaction_method": "silence"},
    )

    assert isinstance(config.redact_pii_audio_options, aai.RedactPiiAudioOptions)
    assert (
        config.redact_pii_audio_options.override_audio_redaction_method
        == aai.PIIRedactedAudioMethod.silence
    )


def test_nested_model_dict_passthrough_also_works_via_constructor():
    """
    The same coercion applies at construction time, not just on later assignment --
    `validate_assignment` covers every `__init__` write too.
    """
    config = aai.TranscriptionConfig(
        language_detection_options={"expected_languages": ["en"]},
        keyterms_prompt_options={"keyterms_match_strength": "standard"},
    )

    assert isinstance(config.language_detection_options, aai.LanguageDetectionOptions)
    assert isinstance(config.keyterms_prompt_options, aai.KeytermsPromptOptions)


def test_malformed_assignment_now_raises():
    """
    Assigning a value pydantic can't coerce now raises immediately, instead of being
    silently stored and failing later/confusingly.
    """
    config = aai.TranscriptionConfig()

    with pytest.raises(ValueError):
        config.temperature = "not-a-number"


def test_set_word_boost_valid_still_works():
    config = aai.TranscriptionConfig().set_word_boost(
        ["AssemblyAI"], boost=aai.WordBoost.high
    )

    assert config.word_boost == ["AssemblyAI"]
    assert config.boost_param == aai.WordBoost.high


def test_set_word_boost_default_boost_still_works():
    config = aai.TranscriptionConfig().set_word_boost(["AssemblyAI"])

    assert config.word_boost == ["AssemblyAI"]
    assert config.boost_param == aai.WordBoost.default


def test_set_word_boost_invalid_boost_now_raises():
    config = aai.TranscriptionConfig()

    with pytest.raises(ValueError):
        config.set_word_boost(["AssemblyAI"], boost="not-a-real-boost-level")


def test_set_custom_spelling_merge_semantics_preserved():
    """
    `set_custom_spelling` now reassigns `custom_spelling` instead of mutating it in
    place, so `validate_assignment` covers it too. Confirm merge (override=False)
    still accumulates across calls rather than overwriting.
    """
    config = aai.TranscriptionConfig()

    config.set_custom_spelling({"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling({"Kubernetes": "k8s"}, override=False)

    assert config.custom_spelling == {
        "AssemblyAI": ["assemblyAI"],
        "Kubernetes": ["k8s"],
    }


def test_set_custom_spelling_override_still_replaces():
    config = aai.TranscriptionConfig()

    config.set_custom_spelling({"AssemblyAI": "assemblyAI"})
    config.set_custom_spelling({"Kubernetes": "k8s"}, override=True)

    assert config.custom_spelling == {"Kubernetes": ["k8s"]}
