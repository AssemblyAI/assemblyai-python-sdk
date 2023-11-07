import factory
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class EntityFactory(factory.Factory):
    class Meta:
        model = aai.types.Entity

    entity_type = factory.Faker("enum", enum_cls=aai.types.EntityType)
    text = factory.Faker("sentence")
    start = factory.Faker("pyint")
    end = factory.Faker("pyint")


class EntityDetectionResponseFactory(factories.TranscriptCompletedResponseFactory):
    entities = factory.List([factory.SubFactory(EntityFactory)])


def test_entity_detection_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `entity_detection` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("entity_detection") is None
    assert transcript.entities is None


def test_entity_detection_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `entity_detection=True` in the `TranscriptionConfig`
    will result in `entity_detection=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(EntityDetectionResponseFactory)()
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(entity_detection=True),
    )

    # Check that request body was properly defined
    assert request_body.get("entity_detection") is True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.entities is not None
    assert len(transcript.entities) > 0
    assert len(transcript.entities) == len(mock_response["entities"])

    for entity in transcript.entities:
        assert len(entity.text.strip()) > 0
