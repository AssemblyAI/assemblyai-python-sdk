import json
from typing import Any, Dict, Tuple

import factory
import httpx
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories

aai.settings.api_key = "test"


class IABLabelResultFactory(factory.Factory):
    class Meta:
        model = aai.types.IABLabelResult

    relevance = factory.Faker("pyfloat", min_value=0, max_value=1)
    label = factory.Faker("word")


class IABResultFactory(factory.Factory):
    class Meta:
        model = aai.types.IABResult

    text = factory.Faker("sentence")
    labels = factory.List([factory.SubFactory(IABLabelResultFactory)])
    timestamp = factory.SubFactory(factories.TimestampFactory)


class IABResponseFactory(factory.Factory):
    class Meta:
        model = aai.types.IABResponse

    status = aai.types.StatusResult.success.value
    results = factory.List([factory.SubFactory(IABResultFactory)])
    summary = factory.Dict(
        {
            "Automotive>AutoType>ConceptCars": factory.Faker(
                "pyfloat", min_value=0, max_value=1
            )
        }
    )


class IABCategoriesResponseFactory(factories.TranscriptCompletedResponseFactory):
    iab_categories_result = factory.SubFactory(IABResponseFactory)


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
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}",
        status_code=httpx.codes.OK,
        method="POST",
        json={
            **mock_processing_response,
            "id": mock_transcript_id,  # inject ID from main mock response
        },
    )

    # Mock polling-for-completeness response, with completed transcript
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{mock_transcript_id}",
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


def test_iab_categories_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `iab_categories` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """

    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("iab_categories") is None
    assert transcript.iab_categories is None


def test_iab_categories_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `iab_categories=True` in the `TranscriptionConfig` will
    result in `iab_categories` being included in the request body, and that
    the response will be properly parsed into the `Transcript` object
    """

    mock_response = factories.generate_dict_factory(IABCategoriesResponseFactory)()

    request_body, transcript = __submit_mock_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(iab_categories=True),
    )

    assert request_body.get("iab_categories") is True

    assert transcript.error is None

    assert transcript.iab_categories is not None
    assert transcript.iab_categories.status == mock_response.get(
        "iab_categories_result", {}
    ).get("status")

    # Validate results
    response_results = mock_response.get("iab_categories_result", {}).get("results", [])
    transcript_results = transcript.iab_categories.results

    assert transcript_results is not None
    assert len(transcript_results) == len(response_results)
    assert len(transcript_results) > 0

    for response_result, transcript_result in zip(response_results, transcript_results):
        assert transcript_result.text == response_result.get("text")
        assert len(transcript_result.text) > 0

        assert len(transcript_result.labels) > 0
        assert len(transcript_result.labels) == len(response_result.get("labels", []))
        for response_label, transcript_label in zip(
            response_result.get("labels", []), transcript_result.labels
        ):
            assert transcript_label.relevance == response_label.get("relevance")
            assert transcript_label.label == response_label.get("label")

    # Validate summary
    response_summary = mock_response.get("iab_categories_result", {}).get("summary", {})
    transcript_summary = transcript.iab_categories.summary

    assert transcript_summary is not None
    assert len(transcript_summary) > 0
    assert transcript_summary == response_summary
