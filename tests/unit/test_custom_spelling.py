import factory
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class CustomSpellingFactory(factory.Factory):
    class Meta:
        model = dict  # The model is a dictionary
        rename = {"_from": "from"}

    _from = factory.List([factory.Faker("word")])  # List of words in 'from'
    to = factory.Faker("word")  # one word in 'to'


class CustomSpellingResponseFactory(factories.TranscriptCompletedResponseFactory):
    @factory.lazy_attribute
    def custom_spelling(self):
        return [CustomSpellingFactory()]


def test_custom_spelling_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that not calling `set_custom_spelling()` on the `TranscriptionConfig`
    will result in the default behavior of it being excluded from the request body.
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("custom_spelling") is None
    assert transcript.json_response.get("custom_spelling") is None


def test_custom_spelling_set_config_succeeds():
    """
    Tests that calling `set_custom_spelling()` on the `TranscriptionConfig`
    will set the values correctly, and that the config values can be accessed again
    through the custom_spelling property.
    """
    config = aai.TranscriptionConfig()

    # Setting a string will be put in a list
    config.set_custom_spelling({"AssemblyAI": "assemblyAI"})
    assert config.custom_spelling == {"AssemblyAI": ["assemblyAI"]}

    # Setting multiple pairs works
    config.set_custom_spelling(
        {"AssemblyAI": "assemblyAI", "Kubernetes": ["k8s", "kubernetes"]}, override=True
    )
    assert config.custom_spelling == {
        "AssemblyAI": ["assemblyAI"],
        "Kubernetes": ["k8s", "kubernetes"],
    }


def test_custom_spelling_enabled(httpx_mock: HTTPXMock):
    """
    Tests that calling `set_custom_spelling()` on the `TranscriptionConfig`
    will result in correct `custom_spelling` in the request body, and that the
    response is properly parsed into the `custom_spelling` field.
    """

    mock_response = factories.generate_dict_factory(CustomSpellingResponseFactory)()

    # Set up the custom spelling config based on the mocked values
    from_ = mock_response["custom_spelling"][0]["from"]
    to = mock_response["custom_spelling"][0]["to"]

    config = aai.TranscriptionConfig().set_custom_spelling({to: from_})

    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=config,
    )

    # Check that request body was properly defined
    custom_spelling_response = request_body["custom_spelling"]
    assert custom_spelling_response is not None and len(custom_spelling_response) > 0
    assert "from" in custom_spelling_response[0]
    assert "to" in custom_spelling_response[0]

    # Check that transcript has no errors and custom spelling response corresponds to request
    assert transcript.error is None
    assert transcript.json_response["custom_spelling"] == custom_spelling_response
