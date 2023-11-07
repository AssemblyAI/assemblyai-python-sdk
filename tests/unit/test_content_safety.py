import random

import factory
import pytest
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class ContentSafetySeverityScoreFactory(factory.Factory):
    class Meta:
        model = aai.types.ContentSafetySeverityScore

    low = factory.Faker("pyfloat")
    medium = factory.Faker("pyfloat")
    high = factory.Faker("pyfloat")


class ContentSafetyLabelResultFactory(factory.Factory):
    class Meta:
        model = aai.types.ContentSafetyLabelResult

    label = factory.Faker("enum", enum_cls=aai.types.ContentSafetyLabel)
    confidence = factory.Faker("pyfloat")
    severity = factory.Faker("pyfloat")


class ContentSafetyResultFactory(factory.Factory):
    class Meta:
        model = aai.types.ContentSafetyResult

    text = factory.Faker("sentence")
    labels = factory.List([factory.SubFactory(ContentSafetyLabelResultFactory)])
    timestamp = factory.SubFactory(factories.TimestampFactory)


class ContentSafetyResponseFactory(factory.Factory):
    class Meta:
        model = aai.types.ContentSafetyResponse

    status = aai.types.StatusResult.success
    results = factory.List([factory.SubFactory(ContentSafetyResultFactory)])
    summary = factory.Dict(
        {
            random.choice(list(aai.types.ContentSafetyLabel)).value: factory.Faker(
                "pyfloat"
            )
        }
    )
    severity_score_summary = factory.Dict(
        {
            random.choice(list(aai.types.ContentSafetyLabel)).value: factory.SubFactory(
                ContentSafetySeverityScoreFactory
            )
        }
    )


class ContentSafetyTranscriptResponseFactory(
    factories.TranscriptCompletedResponseFactory
):
    content_safety_labels = factory.SubFactory(ContentSafetyResponseFactory)


def test_content_safety_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `content_safety` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("content_safety") is None
    assert transcript.content_safety is None


def test_content_safety_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `content_safety=True` in the `TranscriptionConfig`
    will result in `content_safety=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(
        ContentSafetyTranscriptResponseFactory
    )()
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(content_safety=True),
    )

    # Check that request body was properly defined
    assert request_body.get("content_safety") is True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.content_safety is not None

    # Verify status
    assert transcript.content_safety.status == aai.types.StatusResult.success

    # Verify results
    assert transcript.content_safety.results is not None
    assert len(transcript.content_safety.results) > 0
    assert len(transcript.content_safety.results) == len(
        mock_response["content_safety_labels"]["results"]
    )
    for response_result, transcript_result in zip(
        mock_response["content_safety_labels"]["results"],
        transcript.content_safety.results,
    ):
        assert transcript_result.text == response_result["text"]

        assert (
            transcript_result.timestamp.start == response_result["timestamp"]["start"]
        )
        assert transcript_result.timestamp.end == response_result["timestamp"]["end"]

        assert len(transcript_result.labels) > 0
        assert len(transcript_result.labels) == len(response_result["labels"])
        for response_label, transcript_label in zip(
            response_result["labels"], transcript_result.labels
        ):
            assert transcript_label.label == response_label["label"]
            assert transcript_label.confidence == response_label["confidence"]
            assert transcript_label.severity == response_label["severity"]

    # Verify summary
    assert transcript.content_safety.summary is not None
    assert len(transcript.content_safety.summary) > 0
    assert len(transcript.content_safety.summary) == len(
        mock_response["content_safety_labels"]["summary"]
    )
    for response_summary_items, transcript_summary_items in zip(
        mock_response["content_safety_labels"]["summary"].items(),
        transcript.content_safety.summary.items(),
    ):
        response_summary_key, response_summary_value = response_summary_items
        transcript_summary_key, transcript_summary_value = transcript_summary_items

        assert transcript_summary_key == response_summary_key
        assert transcript_summary_value == response_summary_value

    # Verify severity score summary
    assert transcript.content_safety.severity_score_summary is not None
    assert len(transcript.content_safety.severity_score_summary) > 0
    assert len(transcript.content_safety.severity_score_summary) == len(
        mock_response["content_safety_labels"]["severity_score_summary"]
    )
    for (
        response_severity_score_summary_items,
        transcript_severity_score_summary_items,
    ) in zip(
        mock_response["content_safety_labels"]["severity_score_summary"].items(),
        transcript.content_safety.severity_score_summary.items(),
    ):
        (
            response_severity_score_summary_key,
            response_severity_score_summary_values,
        ) = response_severity_score_summary_items
        (
            transcript_severity_score_summary_key,
            transcript_severity_score_summary_values,
        ) = transcript_severity_score_summary_items

        assert (
            transcript_severity_score_summary_key == response_severity_score_summary_key
        )
        assert (
            transcript_severity_score_summary_values.high
            == response_severity_score_summary_values["high"]
        )
        assert (
            transcript_severity_score_summary_values.medium
            == response_severity_score_summary_values["medium"]
        )
        assert (
            transcript_severity_score_summary_values.low
            == response_severity_score_summary_values["low"]
        )


def test_content_safety_with_confidence_threshold(httpx_mock: HTTPXMock):
    """
    Tests that `content_safety_confidence` can be set in the `TranscriptionConfig`
    and will be included in the request body
    """
    confidence = 40
    request, _ = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response={},  # Response doesn't matter here; we're just testing the request body
        config=aai.TranscriptionConfig(
            content_safety=True, content_safety_confidence=confidence
        ),
    )

    assert request.get("content_safety") is True
    assert request.get("content_safety_confidence") == confidence


@pytest.mark.parametrize("confidence", [1, 101])
def test_content_safety_with_invalid_confidence_threshold(
    httpx_mock: HTTPXMock, confidence: int
):
    """
    Tests that a `content_safety_confidence` outside the acceptable range will cause
    an exception to be raised before the request is sent
    """
    with pytest.raises(ValueError) as error:
        unit_test_utils.submit_mock_transcription_request(
            httpx_mock,
            mock_response={},  # We don't expect to produce a response
            config=aai.TranscriptionConfig(
                content_safety=True, content_safety_confidence=confidence
            ),
        )

    assert "content_safety_confidence" in str(error)

    # Check that the error was raised before any requests were made
    assert len(httpx_mock.get_requests()) == 0

    # Inform httpx_mock that it's okay we didn't make any requests
    httpx_mock.reset(False)
