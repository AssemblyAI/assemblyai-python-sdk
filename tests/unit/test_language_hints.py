import copy
import json

import httpx
import pytest

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories

aai.settings.api_key = "test"


@pytest.mark.parametrize(
    "hint",
    [
        None,
        {"default_language": "en"},
        {"default_language": "en", "languages": ["de", "fr", "en"]},
    ],
)
def test_hint_survives_actual_submit_request(httpx_mock, hint):
    response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )
    config = aai.TranscriptionConfig(
        language_hints=hint,
        prompt="Transcribe exactly.",
        language_detection=True,
        language_codes=["fr", "en"],
    )
    aai.Transcriber().submit("https://example.org/audio.wav", config=config)
    payload = json.loads(httpx_mock.get_request().content)
    if hint is None:
        assert "language_hints" not in payload
    else:
        assert payload["language_hints"] == hint
    assert payload["prompt"] == "Transcribe exactly."
    assert payload["language_detection"] is True
    assert payload["language_codes"] == ["fr", "en"]


def test_per_file_config_copies_and_clear():
    shared = aai.TranscriptionConfig(
        language_hints={"default_language": "en", "languages": ["en", "fr"]}
    )
    french = copy.deepcopy(shared)
    french.language_hints = {"default_language": "fr", "languages": ["fr", "en"]}
    assert shared.language_hints.default_language == "en"
    assert shared.language_hints.languages == ["en", "fr"]
    french.language_hints = None
    assert french.language_hints is None
    model = aai.LanguageHints(default_language="en", languages=["de", "en"])
    shared.language_hints = model
    model.languages.append("fr")
    assert shared.language_hints.languages == ["de", "en"]


@pytest.mark.parametrize(
    "hint",
    [
        {},
        {"default_language": ""},
        {"default_language": "English"},
        {"default_language": "en", "languages": []},
        {"default_language": "en", "languages": ["fr"]},
        {"default_language": "en", "languages": ["en", "EN"]},
        {"default_language": "en", "unknown": "value"},
    ],
)
def test_invalid_hint_rejected(hint):
    with pytest.raises(ValueError):
        aai.TranscriptionConfig(language_hints=hint)


def test_raw_configuration_preserves_hints():
    raw = aai.RawTranscriptionConfig(
        language_hints={"default_language": "fr", "languages": ["en", "fr"]}
    )
    config = aai.TranscriptionConfig(raw_transcription_config=raw)
    assert config.language_hints.default_language == "fr"
    assert config.language_hints.languages == ["en", "fr"]
    override = aai.TranscriptionConfig(
        raw_transcription_config=raw, language_hints={"default_language": "de"}
    )
    assert override.language_hints.default_language == "de"
