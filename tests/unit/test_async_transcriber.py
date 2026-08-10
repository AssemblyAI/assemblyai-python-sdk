import asyncio
import io
import json
import os
import re
from urllib.parse import urlencode

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_TRANSCRIPT, ENDPOINT_UPLOAD
from tests.unit import factories

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

TRANSCRIPT_URL = f"{aai.settings.base_url}{ENDPOINT_TRANSCRIPT}"
UPLOAD_URL = f"{aai.settings.base_url}{ENDPOINT_UPLOAD}"


@pytest.fixture(autouse=True)
def fast_polling():
    """Keeps the polling loop from spending seconds sleeping in tests."""

    original = aai.settings.polling_interval
    aai.settings.polling_interval = 0.001
    yield
    aai.settings.polling_interval = original


def _completed_response(**overrides) -> dict:
    response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    response.update(overrides)

    return response


def _processing_response(transcript_id: str) -> dict:
    response = factories.generate_dict_factory(
        factories.TranscriptProcessingResponseFactory
    )()
    response["id"] = transcript_id

    return response


def _mock_submit(httpx_mock: HTTPXMock, response: dict) -> None:
    httpx_mock.add_response(
        url=TRANSCRIPT_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )


def _mock_poll(httpx_mock: HTTPXMock, response: dict, **kwargs) -> None:
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{response['id']}",
        method="GET",
        status_code=httpx.codes.OK,
        json=response,
        **kwargs,
    )


async def test_transcribe_url_submits_and_polls(httpx_mock: HTTPXMock):
    # Given a job that is queued on submission and completed when polled
    completed = _completed_response()
    _mock_submit(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    # When transcribing a URL
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.transcribe("https://example.org/audio.wav")

    # Then the completed transcript is returned
    assert isinstance(transcript, aai.AsyncTranscript)
    assert transcript.id == completed["id"]
    assert transcript.status == aai.TranscriptStatus.completed
    assert transcript.text == completed["text"]
    assert transcript.words is not None
    assert transcript.utterances is not None

    # And the URL was submitted as-is, without an upload
    submission = json.loads(httpx_mock.get_requests()[0].read())
    assert submission["audio_url"] == "https://example.org/audio.wav"
    assert len(httpx_mock.get_requests()) == 2


async def test_transcribe_passes_config(httpx_mock: HTTPXMock):
    # Given a completed job
    completed = _completed_response()
    _mock_submit(httpx_mock, completed)

    # When transcribing with a config
    config = aai.TranscriptionConfig(
        speech_models=["universal-3-5-pro"],
        speaker_labels=True,
    )
    async with aai.AsyncTranscriber(config=config) as transcriber:
        await transcriber.submit("https://example.org/audio.wav")

    # Then the config travels in the submission body
    submission = json.loads(httpx_mock.get_requests()[0].read())
    assert submission["speech_models"] == ["universal-3-5-pro"]
    assert submission["speaker_labels"] is True


async def test_submit_does_not_poll(httpx_mock: HTTPXMock):
    # Given a job that comes back as processing
    transcript_id = "some-id"
    _mock_submit(httpx_mock, _processing_response(transcript_id))

    # When submitting without waiting
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")

    # Then only the submission request was made
    assert transcript.status == aai.TranscriptStatus.processing
    assert len(httpx_mock.get_requests()) == 1


async def test_wait_for_completion_polls_until_terminal(httpx_mock: HTTPXMock):
    # Given a job that stays queued for two polls before completing
    completed = _completed_response()
    _mock_submit(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    # When transcribing
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.transcribe("https://example.org/audio.wav")

    # Then it polled until the status was terminal
    assert transcript.status == aai.TranscriptStatus.completed
    assert len(httpx_mock.get_requests()) == 4


async def test_transcribe_surfaces_error_status(httpx_mock: HTTPXMock):
    # Given a job that fails server-side
    error_response = factories.generate_dict_factory(
        factories.TranscriptErrorResponseFactory
    )()
    error_response["id"] = "error-id"
    _mock_submit(httpx_mock, _processing_response("error-id"))
    _mock_poll(httpx_mock, error_response)

    # When transcribing
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.transcribe("https://example.org/audio.wav")

    # Then the failure surfaces on the transcript rather than as an exception
    assert transcript.status == aai.TranscriptStatus.error
    assert transcript.error == "Aw, snap!"


async def test_submit_raises_on_http_error(httpx_mock: HTTPXMock):
    # Given a submission that is rejected
    httpx_mock.add_response(
        url=TRANSCRIPT_URL,
        method="POST",
        status_code=httpx.codes.BAD_REQUEST,
        json={"error": "something went wrong"},
    )

    # When submitting, then a TranscriptError carries the server message
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(aai.TranscriptError) as exc_info:
            await transcriber.submit("https://example.org/audio.wav")

    assert "something went wrong" in str(exc_info.value)


async def test_upload_file_from_bytes(httpx_mock: HTTPXMock):
    # Given a mocked upload endpoint
    audio = os.urandom(64)
    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
        match_content=audio,
    )

    # When uploading raw bytes
    async with aai.AsyncTranscriber() as transcriber:
        upload_url = await transcriber.upload_file(audio)

    # Then the audio is posted verbatim
    assert upload_url == "https://example.org/uploaded.wav"


async def test_upload_file_from_path_streams_with_content_length(
    httpx_mock: HTTPXMock,
    tmp_path,
):
    # Given a local audio file larger than one read chunk
    audio = os.urandom(3 * 1024 * 1024)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(audio)

    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
    )

    # When uploading it by path
    async with aai.AsyncTranscriber() as transcriber:
        upload_url = await transcriber.upload_file(str(audio_path))

    # Then the whole file arrived, sized rather than chunk-encoded
    assert upload_url == "https://example.org/uploaded.wav"
    request = httpx_mock.get_requests()[0]
    assert request.read() == audio
    assert request.headers["content-length"] == str(len(audio))
    assert "transfer-encoding" not in request.headers


async def test_upload_file_from_file_object(httpx_mock: HTTPXMock):
    # Given an in-memory binary file object
    audio = os.urandom(128)
    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
    )

    # When uploading it
    async with aai.AsyncTranscriber() as transcriber:
        await transcriber.upload_file(io.BytesIO(audio))

    # Then its length is known up front and the bytes arrive intact
    request = httpx_mock.get_requests()[0]
    assert request.read() == audio
    assert request.headers["content-length"] == str(len(audio))


async def test_upload_file_from_partially_read_file_object(httpx_mock: HTTPXMock):
    # Given a file object that has already been read from
    audio = os.urandom(128)
    stream = io.BytesIO(audio)
    stream.read(28)

    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
    )

    # When uploading it
    async with aai.AsyncTranscriber() as transcriber:
        await transcriber.upload_file(stream)

    # Then only the remaining bytes are sent, and Content-Length matches them
    request = httpx_mock.get_requests()[0]
    assert request.read() == audio[28:]
    assert request.headers["content-length"] == str(len(audio) - 28)


async def test_upload_file_from_pathlike(httpx_mock: HTTPXMock, tmp_path):
    # Given a local file addressed by a `pathlib.Path`
    audio = os.urandom(64)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(audio)

    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
    )

    # When uploading it
    async with aai.AsyncTranscriber() as transcriber:
        await transcriber.upload_file(audio_path)

    assert httpx_mock.get_requests()[0].read() == audio


async def test_upload_file_rejects_unsupported_input():
    # When uploading something that is neither bytes, a path, nor a file object
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(TypeError):
            await transcriber.upload_file(42)  # type: ignore[arg-type]


async def test_transcribe_local_file_uploads_then_submits(
    httpx_mock: HTTPXMock,
    tmp_path,
):
    # Given a local file and a completed job
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(os.urandom(32))

    completed = _completed_response()
    httpx_mock.add_response(
        url=UPLOAD_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"upload_url": "https://example.org/uploaded.wav"},
    )
    _mock_submit(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    # When transcribing the local file
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.transcribe(str(audio_path))

    # Then the uploaded URL is what gets submitted
    assert transcript.status == aai.TranscriptStatus.completed
    submission = json.loads(httpx_mock.get_requests()[1].read())
    assert submission["audio_url"] == "https://example.org/uploaded.wav"


async def test_get_sentences_and_paragraphs(httpx_mock: HTTPXMock):
    # Given a completed transcript with sentences and paragraphs available
    completed = _completed_response()
    transcript_id = completed["id"]

    sentences = factories.generate_dict_factory(factories.SentencesResponseFactory)()
    paragraphs = factories.generate_dict_factory(factories.ParagraphsResponseFactory)()

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/sentences",
        method="GET",
        status_code=httpx.codes.OK,
        json=sentences,
    )
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/paragraphs",
        method="GET",
        status_code=httpx.codes.OK,
        json=paragraphs,
    )

    # When requesting both
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")
        got_sentences = await transcript.get_sentences()
        got_paragraphs = await transcript.get_paragraphs()

    # Then they are parsed into the corresponding models
    assert [s.text for s in got_sentences] == [
        s["text"] for s in sentences["sentences"]
    ]
    assert [p.text for p in got_paragraphs] == [
        p["text"] for p in paragraphs["paragraphs"]
    ]


async def test_export_subtitles(httpx_mock: HTTPXMock):
    # Given a completed transcript whose subtitle endpoints return text
    completed = _completed_response()
    transcript_id = completed["id"]

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/srt",
        method="GET",
        status_code=httpx.codes.OK,
        text="srt-subtitles",
    )
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/vtt?chars_per_caption=32",
        method="GET",
        status_code=httpx.codes.OK,
        text="vtt-subtitles",
    )

    # When exporting subtitles, with and without a caption limit
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")
        srt = await transcript.export_subtitles_srt()
        vtt = await transcript.export_subtitles_vtt(chars_per_caption=32)

    assert srt == "srt-subtitles"
    assert vtt == "vtt-subtitles"


async def test_word_search(httpx_mock: HTTPXMock):
    # Given a completed transcript and a word-search result
    completed = _completed_response()
    matches = factories.generate_dict_factory(
        factories.WordSearchMatchResponseFactory
    )()

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{completed['id']}/word-search?{urlencode({'words': 'foo,bar'})}",
        method="GET",
        status_code=httpx.codes.OK,
        json=matches,
    )

    # When searching for words
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")
        found = await transcript.word_search(["foo", "bar"])

    assert [m.text for m in found] == [m["text"] for m in matches["matches"]]


async def test_list_transcripts(httpx_mock: HTTPXMock):
    # Given a page of transcripts
    page = factories.generate_dict_factory(factories.ListTranscriptResponse)()
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(TRANSCRIPT_URL)}\?.*"),
        method="GET",
        status_code=httpx.codes.OK,
        json=page,
    )

    # When listing them with parameters
    async with aai.AsyncTranscriber() as transcriber:
        result = await transcriber.list_transcripts(
            aai.ListTranscriptParameters(limit=2)
        )

    assert isinstance(result, aai.ListTranscriptResponse)
    assert len(result.transcripts) == len(page["transcripts"])
    assert httpx_mock.get_requests()[0].url.params["limit"] == "2"


async def test_get_by_id_waits_for_completion(httpx_mock: HTTPXMock):
    # Given an existing transcript that is still processing
    completed = _completed_response()
    _mock_poll(httpx_mock, _processing_response(completed["id"]))
    _mock_poll(httpx_mock, completed)

    # When fetching it by id
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.get_by_id(completed["id"])

    assert transcript.status == aai.TranscriptStatus.completed
    assert transcript.text == completed["text"]


async def test_delete_by_id(httpx_mock: HTTPXMock):
    # Given a transcript that gets deleted
    deleted = factories.generate_dict_factory(
        factories.TranscriptDeletedResponseFactory
    )()
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{deleted['id']}",
        method="DELETE",
        status_code=httpx.codes.OK,
        json=deleted,
    )

    # When deleting it
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.delete_by_id(deleted["id"])

    assert transcript.id == deleted["id"]
    assert transcript.text == "Deleted by user."


async def test_transcribe_group_preserves_order(httpx_mock: HTTPXMock):
    # Given three URLs whose jobs complete out of order
    urls = [f"https://example.org/{i}.wav" for i in range(3)]
    completed = [_completed_response(text=f"transcript {i}") for i in range(3)]

    def submit_callback(request: httpx.Request) -> httpx.Response:
        index = urls.index(json.loads(request.read())["audio_url"])

        return httpx.Response(httpx.codes.OK, json=completed[index])

    httpx_mock.add_callback(
        submit_callback,
        url=TRANSCRIPT_URL,
        method="POST",
        is_reusable=True,
    )

    # When transcribing them as a group
    async with aai.AsyncTranscriber() as transcriber:
        transcripts = await transcriber.submit_group(urls)

    # Then the results line up with the input order
    assert [t.text for t in transcripts] == [
        "transcript 0",
        "transcript 1",
        "transcript 2",
    ]


async def test_transcribe_group_raises_first_failure(httpx_mock: HTTPXMock):
    # Given a batch in which every submission is rejected
    httpx_mock.add_response(
        url=TRANSCRIPT_URL,
        method="POST",
        status_code=httpx.codes.BAD_REQUEST,
        json={"error": "nope"},
        is_reusable=True,
    )

    # When submitting the group without asking for failures
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(aai.TranscriptError):
            await transcriber.submit_group(
                ["https://example.org/a.wav", "https://example.org/b.wav"]
            )


async def test_transcribe_group_returns_failures(httpx_mock: HTTPXMock):
    # Given a batch where one submission fails and one succeeds
    completed = _completed_response(text="ok")

    def submit_callback(request: httpx.Request) -> httpx.Response:
        if json.loads(request.read())["audio_url"].endswith("bad.wav"):
            return httpx.Response(httpx.codes.BAD_REQUEST, json={"error": "nope"})

        return httpx.Response(httpx.codes.OK, json=completed)

    httpx_mock.add_callback(
        submit_callback,
        url=TRANSCRIPT_URL,
        method="POST",
        is_reusable=True,
    )

    # When submitting with return_failures
    async with aai.AsyncTranscriber() as transcriber:
        transcripts, failures = await transcriber.submit_group(
            ["https://example.org/bad.wav", "https://example.org/good.wav"],
            return_failures=True,
        )

    # Then the successful transcript and the error come back side by side
    assert [t.text for t in transcripts] == ["ok"]
    assert len(failures) == 1
    assert isinstance(failures[0], aai.TranscriptError)


async def test_transcribe_group_limits_concurrency(httpx_mock: HTTPXMock):
    # Given a batch of six jobs and a mock that tracks in-flight requests
    in_flight = 0
    peak = 0
    completed = _completed_response()

    async def submit_callback(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
        finally:
            in_flight -= 1

        return httpx.Response(httpx.codes.OK, json=completed)

    httpx_mock.add_callback(
        submit_callback,
        url=TRANSCRIPT_URL,
        method="POST",
        is_reusable=True,
    )

    # When submitting them with a concurrency limit of two
    async with aai.AsyncTranscriber() as transcriber:
        transcripts = await transcriber.submit_group(
            [f"https://example.org/{i}.wav" for i in range(6)],
            max_concurrency=2,
        )

    # Then no more than two were ever in flight, and all six completed
    assert len(transcripts) == 6
    assert peak == 2


async def test_transcribe_group_rejects_invalid_concurrency():
    async with aai.AsyncTranscriber() as transcriber:
        with pytest.raises(ValueError):
            await transcriber.submit_group(
                ["https://example.org/a.wav"], max_concurrency=0
            )


async def test_transcriptions_run_concurrently(httpx_mock: HTTPXMock):
    # Given a submission endpoint that takes 50ms per request
    completed = _completed_response()

    async def submit_callback(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)

        return httpx.Response(httpx.codes.OK, json=completed)

    httpx_mock.add_callback(
        submit_callback,
        url=TRANSCRIPT_URL,
        method="POST",
        is_reusable=True,
    )

    # When submitting four jobs concurrently on one thread
    async with aai.AsyncTranscriber() as transcriber:
        started = asyncio.get_event_loop().time()
        await asyncio.gather(
            *(transcriber.submit(f"https://example.org/{i}.wav") for i in range(4))
        )
        elapsed = asyncio.get_event_loop().time() - started

    # Then they overlapped instead of running back to back (4 x 50ms)
    assert elapsed < 0.15


async def test_properties_require_a_fetched_response():
    # Given a transcript that has not been fetched
    async with aai.AsyncTranscriber() as transcriber:
        transcript = aai.AsyncTranscript(
            transcript_id="some-id", client=transcriber.client
        )

        # Then reading its fields fails loudly
        with pytest.raises(ValueError):
            transcript.text

        with pytest.raises(ValueError):
            transcript.config

        # But its id is still available
        assert transcript.id == "some-id"


async def test_operations_require_a_transcript_id():
    # Given a transcript with no id at all
    async with aai.AsyncTranscriber() as transcriber:
        transcript = aai.AsyncTranscript(transcript_id=None, client=transcriber.client)

        with pytest.raises(ValueError):
            await transcript.wait_for_completion()

        with pytest.raises(ValueError):
            await transcript.get_sentences()


async def test_config_reflects_submitted_options(httpx_mock: HTTPXMock):
    # Given a transcript that was created with speaker labels
    completed = _completed_response(speaker_labels=True)
    _mock_submit(httpx_mock, completed)

    # When reading the config back off the transcript
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")

    assert transcript.config.speaker_labels is True


async def test_redacted_audio_polls_until_ready(httpx_mock: HTTPXMock):
    # Given a transcript with PII audio redaction enabled
    completed = _completed_response(
        redact_pii=True,
        redact_pii_audio=True,
        redact_pii_policies=["person_name"],
    )
    transcript_id = completed["id"]
    redacted_url = "https://example.org/redacted.wav"

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/redacted-audio",
        method="GET",
        status_code=httpx.codes.ACCEPTED,
        json={},
    )
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{transcript_id}/redacted-audio",
        method="GET",
        status_code=httpx.codes.OK,
        json={"status": "redacted_audio_ready", "redacted_audio_url": redacted_url},
    )

    # When asking for the redacted audio URL twice
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")
        first = await transcript.get_redacted_audio_url()
        second = await transcript.get_redacted_audio_url()

    # Then it polled past the 202 and cached the result
    assert first == redacted_url
    assert second == redacted_url
    assert len(httpx_mock.get_requests()) == 3


async def test_redacted_audio_requires_redaction_config(httpx_mock: HTTPXMock):
    # Given a transcript that was not configured for PII audio redaction
    _mock_submit(httpx_mock, _completed_response())

    # Then asking for redacted audio is rejected before any request is made
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")

        with pytest.raises(ValueError):
            await transcript.get_redacted_audio_url()


async def test_save_redacted_audio(httpx_mock: HTTPXMock, tmp_path):
    # Given a transcript whose redacted audio is ready
    completed = _completed_response(
        redact_pii=True,
        redact_pii_audio=True,
        redact_pii_policies=["person_name"],
    )
    redacted_url = "https://example.org/redacted.wav"
    audio = os.urandom(2048)

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{completed['id']}/redacted-audio",
        method="GET",
        status_code=httpx.codes.OK,
        json={"status": "redacted_audio_ready", "redacted_audio_url": redacted_url},
    )
    httpx_mock.add_response(
        url=redacted_url,
        method="GET",
        status_code=httpx.codes.OK,
        content=audio,
    )

    # When saving it to disk
    target = tmp_path / "redacted.wav"
    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")
        await transcript.save_redacted_audio(str(target))

    assert target.read_bytes() == audio


async def test_save_redacted_audio_raises_when_unavailable(
    httpx_mock: HTTPXMock,
    tmp_path,
):
    # Given a redacted audio URL that no longer serves the file
    completed = _completed_response(
        redact_pii=True,
        redact_pii_audio=True,
        redact_pii_policies=["person_name"],
    )
    redacted_url = "https://example.org/redacted.wav"

    _mock_submit(httpx_mock, completed)
    httpx_mock.add_response(
        url=f"{TRANSCRIPT_URL}/{completed['id']}/redacted-audio",
        method="GET",
        status_code=httpx.codes.OK,
        json={"status": "redacted_audio_ready", "redacted_audio_url": redacted_url},
    )
    httpx_mock.add_response(
        url=redacted_url,
        method="GET",
        status_code=httpx.codes.NOT_FOUND,
    )

    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")

        with pytest.raises(aai.types.RedactedAudioUnavailableError):
            await transcript.save_redacted_audio(str(tmp_path / "redacted.wav"))


async def test_sends_authorization_and_user_agent(httpx_mock: HTTPXMock):
    # Given any request
    _mock_submit(httpx_mock, _completed_response())

    async with aai.AsyncTranscriber() as transcriber:
        await transcriber.submit("https://example.org/audio.wav")

    # Then it carries the same auth and user-agent as the sync client
    request = httpx_mock.get_requests()[0]
    assert request.headers["authorization"] == "test"
    assert "AssemblyAI/1.0" in request.headers["user-agent"]


async def test_client_is_closed_when_owned(httpx_mock: HTTPXMock):
    # Given a transcriber that created its own client
    async with aai.AsyncTranscriber() as transcriber:
        client = transcriber.client

    # Then leaving the context closed the pool
    assert client.http_client.is_closed


async def test_supplied_client_is_not_closed():
    # Given a transcriber handed an explicit client
    client = aai.AsyncClient(settings=aai.settings)

    async with aai.AsyncTranscriber(client=client) as transcriber:
        assert transcriber.client is client

    # Then closing the transcriber leaves the caller's pool open
    assert not client.http_client.is_closed
    await client.aclose()
    assert client.http_client.is_closed


async def test_client_records_last_response(httpx_mock: HTTPXMock):
    # Given a completed submission
    _mock_submit(httpx_mock, _completed_response())

    async with aai.AsyncTranscriber() as transcriber:
        assert transcriber.client.last_response is None
        await transcriber.submit("https://example.org/audio.wav")

        # Then the client exposes the last response, like the sync client does
        assert transcriber.client.last_response is not None
        assert transcriber.client.last_response.status_code == httpx.codes.OK


async def test_client_requires_an_api_key():
    # Given settings without an API key
    settings = aai.Settings(api_key=None)

    with pytest.raises(ValueError):
        aai.AsyncClient(settings=settings)


async def test_async_transcript_is_a_lemur_source(httpx_mock: HTTPXMock):
    # Given a completed async transcript
    completed = _completed_response()
    _mock_submit(httpx_mock, completed)

    async with aai.AsyncTranscriber() as transcriber:
        transcript = await transcriber.submit("https://example.org/audio.wav")

        # Then it can be handed to LeMUR, which only needs its id
        source = aai.LemurSource(transcript)

    assert source.source.id == completed["id"]
