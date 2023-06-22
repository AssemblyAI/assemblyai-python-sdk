import uuid

import assemblyai as aai
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
