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


def test_malformed_nested_dict_raises_at_assignment():
    """
    A dict whose *inner* level is wrong fails at assignment too, not just at the top level --
    the error surfaces where the caller wrote it rather than at serialization time.
    """
    config = aai.TranscriptionConfig()

    with pytest.raises(ValueError):
        config.speech_understanding = {"request": {"speaker_identification": "oops"}}

    assert config.speech_understanding is None


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


# Equivalence: the dict form is an additional way to spell the nested-model construction, not a
# replacement. Every pair below builds the same option two ways -- the pre-existing nested-model
# style and the new dict style -- and asserts the resulting config, and the payload it serializes
# to, are indistinguishable.


def _nested_speech_understanding():
    return aai.TranscriptionConfig(
        speech_understanding=aai.SpeechUnderstandingRequest(
            request=aai.SpeechUnderstandingFeatureRequests(
                speaker_identification=aai.SpeakerIdentificationRequest(
                    speaker_type=aai.SpeakerType.name,
                    known_values=["Michel Martin", "Peter DeCarlo"],
                )
            )
        )
    )


def _dict_speech_understanding():
    return aai.TranscriptionConfig(
        speech_understanding={
            "request": {
                "speaker_identification": {
                    "speaker_type": "name",
                    "known_values": ["Michel Martin", "Peter DeCarlo"],
                }
            }
        }
    )


def _dict_speech_understanding_by_assignment():
    config = aai.TranscriptionConfig()
    config.speech_understanding = {
        "request": {
            "speaker_identification": {
                "speaker_type": "name",
                "known_values": ["Michel Martin", "Peter DeCarlo"],
            }
        }
    }
    return config


def _nested_language_detection_options():
    return aai.TranscriptionConfig(
        language_detection_options=aai.LanguageDetectionOptions(
            expected_languages=["en", "es"],
            fallback_language="en",
        )
    )


def _dict_language_detection_options():
    return aai.TranscriptionConfig(
        language_detection_options={
            "expected_languages": ["en", "es"],
            "fallback_language": "en",
        }
    )


def _nested_keyterms_prompt_options():
    config = aai.TranscriptionConfig()
    config.keyterms_prompt_options = aai.KeytermsPromptOptions(
        keyterms_match_strength="high"
    )
    return config


def _dict_keyterms_prompt_options():
    config = aai.TranscriptionConfig()
    config.keyterms_prompt_options = {"keyterms_match_strength": "high"}
    return config


def _nested_speaker_options():
    return aai.TranscriptionConfig().set_speaker_diarization(
        speaker_options=aai.SpeakerOptions(
            min_speakers_expected=2, max_speakers_expected=4
        )
    )


def _dict_speaker_options():
    return aai.TranscriptionConfig().set_speaker_diarization(
        speaker_options={"min_speakers_expected": 2, "max_speakers_expected": 4}
    )


def _nested_redact_pii_audio_options():
    return aai.TranscriptionConfig().set_redact_pii(
        policies=[aai.PIIRedactionPolicy.email_address],
        redact_audio_options=aai.RedactPiiAudioOptions(
            override_audio_redaction_method=aai.PIIRedactedAudioMethod.silence
        ),
    )


def _dict_redact_pii_audio_options():
    return aai.TranscriptionConfig().set_redact_pii(
        policies=[aai.PIIRedactionPolicy.email_address],
        redact_audio_options={"override_audio_redaction_method": "silence"},
    )


EQUIVALENT_CONFIGS = [
    pytest.param(
        "speech_understanding",
        aai.SpeechUnderstandingRequest,
        _nested_speech_understanding,
        _dict_speech_understanding,
        id="speech_understanding-constructor",
    ),
    pytest.param(
        "speech_understanding",
        aai.SpeechUnderstandingRequest,
        _nested_speech_understanding,
        _dict_speech_understanding_by_assignment,
        id="speech_understanding-assignment",
    ),
    pytest.param(
        "language_detection_options",
        aai.LanguageDetectionOptions,
        _nested_language_detection_options,
        _dict_language_detection_options,
        id="language_detection_options",
    ),
    pytest.param(
        "keyterms_prompt_options",
        aai.KeytermsPromptOptions,
        _nested_keyterms_prompt_options,
        _dict_keyterms_prompt_options,
        id="keyterms_prompt_options",
    ),
    pytest.param(
        "speaker_options",
        aai.SpeakerOptions,
        _nested_speaker_options,
        _dict_speaker_options,
        id="speaker_options",
    ),
    pytest.param(
        "redact_pii_audio_options",
        aai.RedactPiiAudioOptions,
        _nested_redact_pii_audio_options,
        _dict_redact_pii_audio_options,
        id="redact_pii_audio_options",
    ),
]


@pytest.mark.parametrize("field, model_type, nested_fn, dict_fn", EQUIVALENT_CONFIGS)
def test_nested_model_and_dict_styles_produce_the_same_value(
    field, model_type, nested_fn, dict_fn
):
    """Both spellings land on the same typed model, holding the same data."""
    nested_value = getattr(nested_fn(), field)
    dict_value = getattr(dict_fn(), field)

    assert isinstance(nested_value, model_type)
    assert isinstance(dict_value, model_type)
    assert type(dict_value) is type(nested_value)
    assert dict_value == nested_value


@pytest.mark.parametrize("field, model_type, nested_fn, dict_fn", EQUIVALENT_CONFIGS)
def test_nested_model_and_dict_styles_serialize_identically(
    field, model_type, nested_fn, dict_fn
):
    """The request body sent to the API is byte-for-byte the same either way."""
    nested_payload = nested_fn().raw.dict(exclude_none=True)
    dict_payload = dict_fn().raw.dict(exclude_none=True)

    assert dict_payload == nested_payload


def test_nested_model_style_is_untouched_by_the_dict_coercion():
    """
    A caller who passes a fully built model gets that model back, not a copy round-tripped
    through a dict -- existing code keeps its exact behavior.
    """
    speech_understanding = aai.SpeechUnderstandingRequest(
        request=aai.SpeechUnderstandingFeatureRequests(
            speaker_identification=aai.SpeakerIdentificationRequest(
                speaker_type=aai.SpeakerType.name,
                known_values=["Michel Martin"],
            )
        )
    )

    config = aai.TranscriptionConfig(speech_understanding=speech_understanding)

    assert config.speech_understanding == speech_understanding
    assert isinstance(
        config.speech_understanding.request.speaker_identification,
        aai.SpeakerIdentificationRequest,
    )
