import uuid

import httpx
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT
from tests.unit import factories


def test_transcript_group_accepts_transcript_ids():
    """
    Tests whether a TranscriptGroup accepts transcript IDs.
    """
    transcript_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    transcript_group = aai.TranscriptGroup(transcript_ids=transcript_ids)

    assert [transcript.id for transcript in transcript_group] == transcript_ids


def test_transcript_group_check_status():
    """
    Tests the TranscriptGroup's status
    """

    # create a mock response of a completed transcript
    mock_completed_transcript = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()

    mock_queued_transcript = factories.generate_dict_factory(
        factories.TranscriptQueuedResponseFactory
    )()

    mock_processing_transcript = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()

    mock_error_transcript = factories.generate_dict_factory(
        factories.TranscriptErrorResponseFactory
    )()

    transcript_completed = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_completed_transcript),
    )

    transcript_queued = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_queued_transcript),
    )

    transcript_processing = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_processing_transcript),
    )

    transcript_error = aai.Transcript.from_response(
        client=aai.Client.get_default(),
        response=aai.types.TranscriptResponse(**mock_error_transcript),
    )

    transcript_group = aai.TranscriptGroup()

    transcript_group.add_transcript(transcript_completed)
    assert transcript_group.status == aai.TranscriptStatus.completed

    transcript_group.add_transcript(transcript_queued)
    assert transcript_group.status == aai.TranscriptStatus.queued

    transcript_group.add_transcript(transcript_processing)
    assert transcript_group.status == aai.TranscriptStatus.queued

    transcript_group.add_transcript(transcript_error)
    assert transcript_group.status == aai.TranscriptStatus.error


def test_get_by_ids(httpx_mock: HTTPXMock):
    transcript_ids = ["123", "456"]
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    for transcript_id in transcript_ids:
        httpx_mock.add_response(
            url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{transcript_id}",
            status_code=httpx.codes.OK,
            method="GET",
            json=mock_transcript_response,
        )

    transcript_group = aai.TranscriptGroup.get_by_ids(transcript_ids)

    assert isinstance(transcript_group, aai.TranscriptGroup)
    assert transcript_group.status == aai.TranscriptStatus.completed
    for transcript in transcript_group:
        assert transcript.id in transcript_ids
        transcript_ids.remove(transcript.id)

        assert transcript.error is None
    assert len(transcript_ids) == 0


_ORDERED_TRANSCRIPT_IDS = [f"transcript-{index}" for index in range(8)]


def _mock_completed_transcripts(httpx_mock: HTTPXMock, failing_id=None):
    """
    Mocks a completed transcript for each of `_ORDERED_TRANSCRIPT_IDS`, and a
    500 for `failing_id`.
    """
    for transcript_id in _ORDERED_TRANSCRIPT_IDS:
        url = f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{transcript_id}"
        if transcript_id == failing_id:
            httpx_mock.add_response(
                url=url,
                status_code=httpx.codes.INTERNAL_SERVER_ERROR,
                method="GET",
                json={"error": "something went wrong"},
            )
            continue

        response = factories.generate_dict_factory(
            factories.TranscriptCompletedResponseFactory
        )()
        response["id"] = transcript_id
        httpx_mock.add_response(
            url=url,
            status_code=httpx.codes.OK,
            method="GET",
            json=response,
        )


def test_get_by_ids_keeps_given_order(
    httpx_mock: HTTPXMock, futures_hash_in_descending_order
):
    """
    Tests that the transcripts of a fetched group are in the order of the
    given IDs.
    """
    _mock_completed_transcripts(httpx_mock)

    transcript_group = aai.TranscriptGroup.get_by_ids(_ORDERED_TRANSCRIPT_IDS)

    assert [t.id for t in transcript_group] == _ORDERED_TRANSCRIPT_IDS


def test_wait_for_completion_keeps_order_when_one_fails(
    httpx_mock: HTTPXMock, futures_hash_in_descending_order
):
    """
    Tests that when polling one transcript fails, it leaves the group and the
    others stay in their order.
    """
    failing_id = _ORDERED_TRANSCRIPT_IDS[3]
    _mock_completed_transcripts(httpx_mock, failing_id=failing_id)

    transcript_group, failures = aai.TranscriptGroup(
        transcript_ids=_ORDERED_TRANSCRIPT_IDS
    ).wait_for_completion(return_failures=True)

    assert [t.id for t in transcript_group] == [
        transcript_id
        for transcript_id in _ORDERED_TRANSCRIPT_IDS
        if transcript_id != failing_id
    ]
    assert len(failures) == 1
    assert failures[0].status_code == httpx.codes.INTERNAL_SERVER_ERROR


def test_get_by_id_async(httpx_mock: HTTPXMock):
    transcript_ids = ["123", "456"]
    mock_transcript_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    for transcript_id in transcript_ids:
        httpx_mock.add_response(
            url=f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}/{transcript_id}",
            status_code=httpx.codes.OK,
            method="GET",
            json=mock_transcript_response,
        )

    transcript_group_future = aai.TranscriptGroup.get_by_ids_async(transcript_ids)
    transcript_group = transcript_group_future.result()

    assert isinstance(transcript_group, aai.TranscriptGroup)
    assert transcript_group.status == aai.TranscriptStatus.completed
    for transcript in transcript_group:
        assert transcript.id in transcript_ids
        transcript_ids.remove(transcript.id)

        assert transcript.error is None
    assert len(transcript_ids) == 0
