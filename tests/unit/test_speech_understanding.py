import assemblyai as aai


def test_speech_understanding_effort_enum():
    assert aai.SpeechUnderstandingEffort.low == "low"
    assert aai.SpeechUnderstandingEffort.medium == "medium"
    assert aai.SpeechUnderstandingEffort.low.value == "low"
    assert aai.SpeechUnderstandingEffort.medium.value == "medium"


def test_speaker_identification_request_effort_enum():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.name,
        effort=aai.SpeechUnderstandingEffort.medium,
    )
    assert req.speaker_type == aai.SpeakerType.name
    assert req.effort == aai.SpeechUnderstandingEffort.medium
    data = req.dict(exclude_none=True)
    assert data["effort"] == "medium"
    assert data["speaker_type"] == "name"


def test_speaker_identification_request_effort_str():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.role,
        known_values=["Agent", "Customer"],
        effort="low",
    )
    assert req.speaker_type == aai.SpeakerType.role
    assert req.known_values == ["Agent", "Customer"]
    assert req.effort == "low"
    data = req.dict(exclude_none=True)
    assert data["effort"] == "low"


def test_speaker_identification_request_effort_omitted():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.name,
    )
    assert req.effort is None
    data = req.dict(exclude_none=True)
    assert "effort" not in data


def test_translation_request_effort_and_force_translation():
    req = aai.TranslationRequest(
        target_languages=["es", "fr"],
        force_translation=True,
        effort=aai.SpeechUnderstandingEffort.medium,
    )
    assert req.target_languages == ["es", "fr"]
    assert req.force_translation is True
    assert req.effort == aai.SpeechUnderstandingEffort.medium
    assert req.formal is False
    assert req.match_original_utterance is False

    data = req.dict(exclude_none=True)
    assert data["target_languages"] == ["es", "fr"]
    assert data["force_translation"] is True
    assert data["effort"] == "medium"


def test_translation_request_defaults():
    req = aai.TranslationRequest(target_languages=["de"])
    assert req.force_translation is False
    assert req.effort is None

    data = req.dict(exclude_none=True)
    assert data["target_languages"] == ["de"]
    assert data["force_translation"] is False
    assert "effort" not in data


def test_custom_formatting_request_effort():
    req = aai.CustomFormattingRequest(
        date="mm/dd/yyyy",
        phone_number="(xxx)xxx-xxxx",
        effort=aai.SpeechUnderstandingEffort.low,
    )
    assert req.date == "mm/dd/yyyy"
    assert req.phone_number == "(xxx)xxx-xxxx"
    assert req.effort == aai.SpeechUnderstandingEffort.low

    data = req.dict(exclude_none=True)
    assert data["date"] == "mm/dd/yyyy"
    assert data["phone_number"] == "(xxx)xxx-xxxx"
    assert data["effort"] == "low"
    assert "email" not in data


def test_speech_understanding_request_nested_effort():
    req = aai.SpeechUnderstandingRequest(
        request=aai.SpeechUnderstandingFeatureRequests(
            speaker_identification=aai.SpeakerIdentificationRequest(
                speaker_type=aai.SpeakerType.name,
                effort=aai.SpeechUnderstandingEffort.medium,
            ),
            translation=aai.TranslationRequest(
                target_languages=["es"],
                force_translation=True,
                effort=aai.SpeechUnderstandingEffort.low,
            ),
            custom_formatting=aai.CustomFormattingRequest(
                date="mm/dd/yyyy",
                effort=aai.SpeechUnderstandingEffort.medium,
            ),
        )
    )

    data = req.dict(exclude_none=True)
    assert data["request"]["speaker_identification"]["effort"] == "medium"
    assert data["request"]["translation"]["effort"] == "low"
    assert data["request"]["translation"]["force_translation"] is True
    assert data["request"]["custom_formatting"]["effort"] == "medium"


def test_transcription_config_speech_understanding_dict_with_effort_and_force_translation():
    config = aai.TranscriptionConfig()
    config.speech_understanding = {
        "request": {
            "speaker_identification": {
                "speaker_type": "name",
                "effort": "medium",
            },
            "translation": {
                "target_languages": ["es"],
                "force_translation": True,
                "effort": "low",
            },
            "custom_formatting": {
                "date": "mm/dd/yyyy",
                "effort": "medium",
            },
        }
    }

    assert isinstance(config.speech_understanding, aai.SpeechUnderstandingRequest)
    features = config.speech_understanding.request
    assert features.speaker_identification.speaker_type == aai.SpeakerType.name
    assert features.speaker_identification.effort == "medium"
    assert features.translation.target_languages == ["es"]
    assert features.translation.force_translation is True
    assert features.translation.effort == "low"
    assert features.custom_formatting.date == "mm/dd/yyyy"
    assert features.custom_formatting.effort == "medium"


def test_speech_understanding_response_parses_request_with_effort_and_force_translation():
    raw_response = {
        "request": {
            "speaker_identification": {
                "speaker_type": "name",
                "effort": "medium",
            },
            "translation": {
                "target_languages": ["es"],
                "force_translation": True,
                "effort": "low",
            },
            "custom_formatting": {
                "date": "mm/dd/yyyy",
                "effort": "medium",
            },
        },
        "response": {
            "speaker_identification": {
                "status": "success",
                "mapping": {"Speaker A": "Alice"},
            },
            "translation": {
                "status": "success",
            },
            "custom_formatting": {
                "status": "success",
                "formatted_text": "Call me at (123)456-7890.",
            },
        },
    }

    res = aai.SpeechUnderstandingResponse.parse_obj(raw_response)
    assert res.request.speaker_identification.effort == "medium"
    assert res.request.translation.effort == "low"
    assert res.request.translation.force_translation is True
    assert res.request.custom_formatting.effort == "medium"
    assert res.response.speaker_identification.status == "success"
    assert res.response.speaker_identification.mapping == {"Speaker A": "Alice"}
