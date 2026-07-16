import factory
from pytest_httpx import HTTPXMock

import tests.unit.unit_test_utils as unit_test_utils
import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


class AutohighlightResultFactory(factory.Factory):
    class Meta:
        model = aai.types.AutohighlightResult

    count = factory.Faker("pyint")
    rank = factory.Faker("pyfloat")
    text = factory.Faker("sentence")
    timestamps = factory.List([factory.SubFactory(factories.TimestampFactory)])


class AutohighlightResponseFactory(factory.Factory):
    class Meta:
        model = aai.types.AutohighlightResponse

    status = aai.types.StatusResult.success
    results = factory.List([factory.SubFactory(AutohighlightResultFactory)])


class AutohighlightTranscriptResponseFactory(
    factories.TranscriptCompletedResponseFactory
):
    auto_highlights_result = factory.SubFactory(AutohighlightResponseFactory)


def test_auto_highlights_disabled_by_default(httpx_mock: HTTPXMock):
    """
    Tests that excluding `auto_highlights` from the `TranscriptionConfig` will
    result in the default behavior of it being excluded from the request body
    """
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )(),
        config=aai.TranscriptionConfig(),
    )
    assert request_body.get("auto_highlights") is None
    assert transcript.auto_highlights is None


def test_auto_highlights_enabled(httpx_mock: HTTPXMock):
    """
    Tests that including `auto_highlights=True` in the `TranscriptionConfig`
    will result in `auto_highlights=True` in the request body, and that the
    response is properly parsed into a `Transcript` object
    """
    mock_response = factories.generate_dict_factory(
        AutohighlightTranscriptResponseFactory
    )()
    request_body, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(auto_highlights=True),
    )

    # Check that request body was properly defined
    assert request_body.get("auto_highlights") is True

    # Check that transcript was properly parsed from JSON response
    assert transcript.error is None
    assert transcript.auto_highlights is not None
    assert (
        transcript.auto_highlights.status
        == mock_response["auto_highlights_result"]["status"]
    )

    assert transcript.auto_highlights.results is not None
    assert len(transcript.auto_highlights.results) > 0
    assert len(transcript.auto_highlights.results) == len(
        mock_response["auto_highlights_result"]["results"]
    )

    for response_result, transcript_result in zip(
        mock_response["auto_highlights_result"]["results"],
        transcript.auto_highlights.results,
    ):
        assert transcript_result.count == response_result["count"]
        assert transcript_result.rank == response_result["rank"]
        assert transcript_result.text == response_result["text"]

        for response_timestamp, transcript_timestamp in zip(
            response_result["timestamps"], transcript_result.timestamps
        ):
            assert transcript_timestamp.start == response_timestamp["start"]
            assert transcript_timestamp.end == response_timestamp["end"]


def test_auto_highlights_parses_result_without_rank(httpx_mock: HTTPXMock):
    """
    Regression for #148. The Auto Highlights API has been observed in
    production to return individual `results` entries with no `rank` field
    set, which previously raised a Pydantic ValidationError and made the
    entire TranscriptResponse unparseable. The field is now Optional, so a
    missing rank parses as None and the rest of the response is preserved.
    """
    mock_response = factories.generate_dict_factory(
        AutohighlightTranscriptResponseFactory
    )()
    # Strip the `rank` from the first highlight to simulate the API's
    # observed behavior. Other fields (count, text, timestamps) are left
    # in place.
    assert mock_response["auto_highlights_result"]["results"], (
        "factory should produce at least one highlight"
    )
    del mock_response["auto_highlights_result"]["results"][0]["rank"]

    _, transcript = unit_test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response=mock_response,
        config=aai.TranscriptionConfig(auto_highlights=True),
    )

    assert transcript.error is None
    assert transcript.auto_highlights is not None
    assert transcript.auto_highlights.results is not None
    assert transcript.auto_highlights.results[0].rank is None
    # The rest of the first result is still present, and other entries
    # are unaffected.
    assert transcript.auto_highlights.results[0].text is not None
    if len(transcript.auto_highlights.results) > 1:
        assert transcript.auto_highlights.results[1].rank is not None
