import json
from typing import Any, Dict, Tuple

import factory
import httpx
from pytest_httpx import HTTPXMock

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


def __submit_mock_request(
    httpx_mock: HTTPXMock,
    mock_response: Dict[str, Any],
    config: aai.TranscriptionConfig,
) -> Tuple[Dict[str, Any], aai.Transcript]:
    """
    Helper function to abstract mock transcriber calls with given `TranscriptionConfig`,
    and perform some common assertions.
    """

    mock_transcript_id = mock_response.get("id", "mock_id")

    # Mock initial submission response (transcript is processing)
    mock_processing_response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript",
        status_code=httpx.codes.OK,
        method="POST",
        json={
            **mock_processing_response,
            "id": mock_transcript_id,  # inject ID from main mock response
        },
    )

    # Mock polling-for-completeness response, with completed transcript
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/transcript/{mock_transcript_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_response,
    )

    # == Make API request via SDK ==
    transcript = aai.Transcriber().transcribe(
        data="https://example.org/audio.wav",
        config=config,
    )

    # Check that submission and polling requests were made
    assert len(httpx_mock.get_requests()) == 2

    # Extract body of initial submission request
    request = httpx_mock.get_requests()[0]
    request_body = json.loads(request.content.decode())

    return request_body, transcript


def test_entity_detection_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `entity_detection` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = __submit_mock_request(
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
    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(entity_detection=True),
    )

    # Check that request body was properly defined
    assert request_body.get("entity_detection") == True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.entities is not None
    assert len(transcript.entities) > 0
    assert len(transcript.entities) == len(mock_response["entities"])

    for entity in transcript.entities:
        assert len(entity.text.strip()) > 0
