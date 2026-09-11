"""Tests for the streamed-upload sync path (`transcribe_live`).

Body-shape assertions go through `httpx.MockTransport`, which consumes the
request stream the way a real transport does. `pytest_httpx` records the
request without reading it on older httpx, so a body read after the call sees a
spent generator or a closed file.
"""

import asyncio
import email
import io
import threading
from typing import List, Optional

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai._multipart import _STREAM_READ_SIZE
from assemblyai.sync.v1 import api, async_api

aai.settings.api_key = "test"

STREAM_URL = f"{aai.settings.sync_base_url}/v1/transcribe/live"

_OK_RESPONSE = {
    "text": "hello world",
    "words": [{"text": "hello", "start": 0, "end": 200, "confidence": 0.9}],
    "confidence": 0.92,
    "audio_duration_ms": 400,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 243.7,
}


def _mock_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=STREAM_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_OK_RESPONSE,
    )


def _chunks(*pieces: bytes):
    for piece in pieces:
        yield piece


def _parts(request: httpx.Request):
    """Parses a recorded request body into its multipart parts, in order."""

    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n"
    message = email.message_from_bytes(raw + request.content)

    return [
        (part.get_param("name", header="content-disposition"), part)
        for part in message.get_payload()
    ]


def _send(chunks, config: Optional[dict] = None, **kwargs) -> List[httpx.Request]:
    """Runs `api.transcribe_live` against a transport that records the body."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(httpx.codes.OK, json=_OK_RESPONSE)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api.transcribe_live(
            client,
            base_url=aai.settings.sync_base_url,
            chunks=chunks,
            filename=kwargs.get("filename", "audio.wav"),
            audio_content_type=kwargs.get("audio_content_type", "audio/wav"),
            model="u3-sync-pro",
            config=config,
            timeout=30.0,
        )

    return seen


async def _asend(chunks, config: Optional[dict] = None) -> List[httpx.Request]:
    """The asyncio counterpart of `_send`."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(httpx.codes.OK, json=_OK_RESPONSE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await async_api.transcribe_live(
            client,
            base_url=aai.settings.sync_base_url,
            chunks=chunks,
            filename="audio.wav",
            audio_content_type="audio/wav",
            model="u3-sync-pro",
            config=config,
            timeout=30.0,
        )

    return seen


# --- body shape -------------------------------------------------------------


def test_body_sends_config_before_audio():
    # Given a config alongside the audio
    # When the body is built
    request = _send(_chunks(b"RIFF", b"fake"), config={"prompt": "a prompt"})[0]

    # Then config precedes audio: the server decodes as bytes land and rejects
    # audio that arrives first
    assert [name for name, _ in _parts(request)] == ["config", "audio"]


def test_body_carries_config_json_and_joined_audio():
    # Given audio produced in several pieces
    # When the body is built
    request = _send(_chunks(b"RIFF", b"fake", b"wav"), config={"prompt": "a prompt"})[0]

    # Then the pieces arrive as one audio part and the config part is JSON
    parts = dict(_parts(request))
    assert parts["audio"].get_payload(decode=True) == b"RIFFfakewav"
    assert parts["config"].get_payload(decode=True) == b'{"prompt": "a prompt"}'


def test_body_sends_an_empty_config_part_when_there_is_no_config():
    # Given no config
    # When the body is built
    request = _send(_chunks(b"RIFF"))[0]

    # Then a config part still precedes the audio, as an empty object: the
    # streaming endpoint rejects audio that no config part came before
    parts = _parts(request)
    assert [name for name, _ in parts] == ["config", "audio"]
    assert dict(parts)["config"].get_payload(decode=True) == b"{}"


def test_body_skips_empty_chunks():
    # Given a producer that yields an empty piece between real ones
    # When the body is built
    request = _send(_chunks(b"RIFF", b"", b"wav"))[0]

    # Then it is dropped rather than ending the chunked body early
    assert dict(_parts(request))["audio"].get_payload(decode=True) == b"RIFFwav"


def test_body_is_uploaded_chunked():
    # Given audio that does not exist yet when the request starts
    # When the body is built
    request = _send(_chunks(b"RIFF", b"more"))[0]

    # Then it is framed without a length, which is what lets the request start
    assert request.headers.get("transfer-encoding") == "chunked"
    assert "content-length" not in request.headers


class _SpyFile(io.BytesIO):
    """A file object that records the size of every read asked of it."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.reads: List[int] = []
        self.threads: List[threading.Thread] = []

    def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        self.threads.append(threading.current_thread())
        return super().read(size)


def test_body_reads_a_file_object_in_bounded_pieces():
    # Given a file object holding more than two read sizes of audio
    payload = b"RIFF" + bytes(2 * _STREAM_READ_SIZE + 100)
    source = _SpyFile(payload)

    # When it is streamed
    request = _send(source, filename="call.wav")[0]

    # Then it is read in fixed pieces until exhausted, never slurped whole
    assert source.reads == [_STREAM_READ_SIZE] * 4
    part = dict(_parts(request))["audio"]
    assert part.get_payload(decode=True) == payload
    assert part.get_filename() == "call.wav"


def test_async_body_reads_a_file_object_off_the_event_loop():
    # Given a file object handed to the async transport
    source = _SpyFile(b"RIFF" + bytes(_STREAM_READ_SIZE + 1))

    # When it is streamed
    request = asyncio.run(_asend(source))[0]

    # Then every read ran in a worker thread, so a blocking read cannot stall
    # the loop, and the body is intact
    assert source.reads == [_STREAM_READ_SIZE] * 3
    assert all(t is not threading.main_thread() for t in source.threads)
    assert dict(_parts(request))["audio"].get_payload(decode=True) == source.getvalue()


def test_body_escapes_the_filename():
    # Given a file whose name would otherwise break the part header
    # When the body is built
    request = _send(_chunks(b"RIFF"), filename='we"ird\r\nname.wav')[0]

    # Then the name is escaped the way httpx escapes its own multipart parts
    # and the audio part still parses
    parts = dict(_parts(request))
    assert parts["audio"].get_filename() == "we%22ird%0D%0Aname.wav"
    assert parts["audio"].get_payload(decode=True) == b"RIFF"


def test_body_accepts_bytearray_and_memoryview_chunks():
    # Given a producer handing out buffer objects rather than bytes
    # When the body is built
    request = _send(iter([bytearray(b"RIFF"), memoryview(b"fake")]))[0]

    # Then they are sent as bytes
    assert dict(_parts(request))["audio"].get_payload(decode=True) == b"RIFFfake"


def test_body_rejects_text_chunks():
    # Given a producer yielding str
    # When the body is built, then the mistake is named rather than failing
    # inside the transport
    with pytest.raises(TypeError, match="binary mode"):
        _send(iter(["RIFF", "fake"]))


def test_body_rejects_a_text_mode_file():
    # Given a file object opened without "b"
    # When the body is built, then the mistake is named
    with pytest.raises(TypeError, match="binary mode"):
        _send(io.StringIO("RIFFfake"))


def test_body_hits_the_stream_endpoint():
    # Given a streamed upload
    # When it is sent
    request = _send(_chunks(b"RIFF"))[0]

    # Then the buffered route is left alone
    assert str(request.url) == STREAM_URL
    assert request.headers["X-AAI-Model"] == "u3-sync-pro"


def test_async_body_matches_its_sync_twin():
    # Given the same audio through the async transport
    async def chunks():
        for piece in (b"RIFF", b"fake", b"wav"):
            yield piece

    # When the body is built
    request = asyncio.run(_asend(chunks(), config={"prompt": "a prompt"}))[0]

    # Then it is framed identically
    parts = dict(_parts(request))
    assert [name for name, _ in _parts(request)] == ["config", "audio"]
    assert parts["audio"].get_payload(decode=True) == b"RIFFfakewav"
    assert request.headers.get("transfer-encoding") == "chunked"


def test_async_body_accepts_a_sync_iterable():
    # Given a plain iterable handed to the async transport
    # When the body is built
    request = asyncio.run(_asend(_chunks(b"RIFF", b"fake")))[0]

    # Then one producer can feed both transports
    assert dict(_parts(request))["audio"].get_payload(decode=True) == b"RIFFfake"


# --- transcriber behaviour --------------------------------------------------


def test_transcribe_live_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked streaming endpoint
    _mock_ok(httpx_mock)

    # When streaming audio chunks
    result = aai.SyncTranscriber().transcribe_live(_chunks(b"RIFF", b"fake"))

    # Then the response is parsed like the buffered path's
    assert isinstance(result, aai.SyncTranscriptResponse)
    assert result.text == "hello world"
    assert result.session_id == _OK_RESPONSE["session_id"]


def test_transcribe_live_sends_model_header(httpx_mock: HTTPXMock):
    # Given a mocked streaming endpoint
    _mock_ok(httpx_mock)

    # When streaming with a model set
    aai.SyncTranscriber().transcribe_live(
        _chunks(b"RIFF"),
        config=aai.SyncTranscriptionConfig(model="u3-sync-pro"),
    )

    # Then the routing header goes out, as on the buffered path
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == STREAM_URL
    assert request.headers["X-AAI-Model"] == "u3-sync-pro"


def test_transcribe_live_uses_the_stream_timeout(httpx_mock: HTTPXMock):
    # Given a mocked streaming endpoint
    _mock_ok(httpx_mock)

    # When streaming audio
    aai.SyncTranscriber().transcribe_live(_chunks(b"RIFF"))

    # Then the request gets the longer budget, which must cover the recording
    # as well as the transcription
    timeout = httpx_mock.get_requests()[0].extensions["timeout"]
    assert timeout["read"] == aai.settings.sync_live_http_timeout


def test_transcribe_live_names_a_file_object(monkeypatch):
    # Given a file object carrying a name
    captured = {}

    def fake(client, **kwargs):
        captured.update(kwargs)
        return aai.SyncTranscriptResponse.parse_obj(_OK_RESPONSE)

    monkeypatch.setattr(api, "transcribe_live", fake)

    stream = io.BytesIO(b"RIFF")
    stream.name = "/tmp/call.wav"

    # When it is streamed
    aai.SyncTranscriber().transcribe_live(stream)

    # Then the audio part takes its name and type from the file
    assert captured["filename"] == "call.wav"
    assert captured["audio_content_type"] == "audio/wav"


def test_transcribe_live_marks_pcm_from_config(monkeypatch):
    # Given a config carrying the fields only raw PCM needs
    captured = {}

    def fake(client, **kwargs):
        captured.update(kwargs)
        return aai.SyncTranscriptResponse.parse_obj(_OK_RESPONSE)

    monkeypatch.setattr(api, "transcribe_live", fake)

    # When streaming
    aai.SyncTranscriber().transcribe_live(
        _chunks(b"\x00\x01"),
        config=aai.SyncTranscriptionConfig(sample_rate=16000, channels=1),
    )

    # Then the audio part selects the PCM decoder
    assert captured["audio_content_type"] == "audio/pcm"
    assert captured["filename"] == "audio.pcm"


def test_transcribe_live_requires_both_pcm_fields():
    # Given a config with only half of what raw PCM needs
    config = aai.SyncTranscriptionConfig(sample_rate=16000)

    # When streaming, then it is rejected before any request is made
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.SyncTranscriber().transcribe_live(_chunks(b"\x00"), config=config)


def test_transcribe_live_rejects_bytes():
    # Given audio the caller already holds whole
    # When streaming it, then it is named as a mistake and sent to transcribe()
    with pytest.raises(TypeError, match="transcribe\\(\\)"):
        aai.SyncTranscriber().transcribe_live(b"RIFFfake-wav-bytes")


def test_transcribe_live_rejects_a_path():
    # Given a path rather than an open file
    # When streaming it, then it is rejected rather than silently opened
    with pytest.raises(TypeError, match="not a path"):
        aai.SyncTranscriber().transcribe_live("./call.wav")


def test_transcribe_live_rejects_an_async_iterable():
    # Given an async producer handed to the synchronous transcriber
    async def chunks():
        yield b"RIFF"

    # When streaming it, then it is turned away before any request is made
    # and pointed at the transcriber that can drive it
    with pytest.raises(TypeError, match="AsyncSyncTranscriber"):
        aai.SyncTranscriber().transcribe_live(chunks())


def test_transcribe_live_rejects_job_api_config():
    # Given the job API's config type
    # When streaming with it, then the mismatch is named
    with pytest.raises(TypeError, match="SyncTranscriptionConfig"):
        aai.SyncTranscriber().transcribe_live(
            _chunks(b"RIFF"),
            config=aai.TranscriptionConfig(),
        )


def test_transcribe_live_raises_on_error_response(httpx_mock: HTTPXMock):
    # Given a rejection the server can send while the upload is still in flight
    httpx_mock.add_response(
        url=STREAM_URL,
        method="POST",
        status_code=httpx.codes.TOO_MANY_REQUESTS,
        json={"status": 429, "title": "Rate Limited", "detail": "slow down"},
        headers={"Retry-After": "3"},
    )

    # When streaming audio
    with pytest.raises(aai.SyncTranscriptError) as exc:
        aai.SyncTranscriber().transcribe_live(_chunks(b"RIFF"))

    # Then it is surfaced with the same envelope parsing as the buffered path
    assert exc.value.status_code == 429
    assert exc.value.error_code == "rate_limited"
    assert exc.value.retry_after == 3


@pytest.mark.asyncio
async def test_async_transcribe_live_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked streaming endpoint
    _mock_ok(httpx_mock)

    async def chunks():
        yield b"RIFF"

    # When streaming from an async producer
    async with aai.AsyncSyncTranscriber() as transcriber:
        result = await transcriber.transcribe_live(chunks())

    # Then the response is parsed like its sync twin's
    assert result.text == "hello world"
    assert str(httpx_mock.get_requests()[0].url) == STREAM_URL


@pytest.mark.asyncio
async def test_async_transcribe_live_rejects_bytes():
    # Given audio the caller already holds whole
    # When streaming it, then the async path names the same mistake
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(TypeError, match="transcribe\\(\\)"):
            await transcriber.transcribe_live(b"RIFF")


@pytest.mark.asyncio
async def test_async_transcribe_live_raises_on_error_response(httpx_mock: HTTPXMock):
    # Given a rejection
    httpx_mock.add_response(
        url=STREAM_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"status": 503, "title": "Capacity Exceeded", "detail": "no capacity"},
    )

    async def chunks():
        yield b"RIFF"

    # When streaming audio, then the error surfaces
    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(aai.SyncTranscriptError) as exc:
            await transcriber.transcribe_live(chunks())

    assert exc.value.error_code == "capacity_exceeded"


# --- push-style sessions ------------------------------------------------------


def _fake_live(monkeypatch, module, drain, error: Optional[Exception] = None) -> dict:
    """
    Replaces the transport with a fake that drains the producer the way a real
    transport would, recording what it saw. `drain` turns the producer into a
    list (sync or async), so an abort raised by the producer propagates from it
    exactly as it would from httpx.
    """
    seen: dict = {"called": threading.Event(), "completed": False}

    if asyncio.iscoroutinefunction(drain):

        async def fake(client, **kwargs):
            seen["called"].set()
            seen["config"] = kwargs["config"]
            seen["chunks"] = await drain(kwargs["chunks"])
            seen["completed"] = True
            if error:
                raise error
            return aai.SyncTranscriptResponse.parse_obj(_OK_RESPONSE)

    else:

        def fake(client, **kwargs):
            seen["called"].set()
            seen["config"] = kwargs["config"]
            seen["chunks"] = drain(kwargs["chunks"])
            seen["completed"] = True
            if error:
                raise error
            return aai.SyncTranscriptResponse.parse_obj(_OK_RESPONSE)

    monkeypatch.setattr(module, "transcribe_live", fake)
    return seen


def _drain(chunks):
    return list(chunks)


async def _adrain(chunks):
    return [chunk async for chunk in chunks]


def test_open_live_uploads_written_chunks_in_order(monkeypatch):
    # Given a session fed from a callback-style producer
    seen = _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live(
            aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
        )
        for piece in (b"\x00\x01", bytearray(b"\x02\x03"), memoryview(b"\x04\x05")):
            session.write(piece)

        # When the audio ends and the result is awaited
        result = session.result()

    # Then the chunks went out in order, as bytes, with the config, and the
    # transcript came back parsed
    assert seen["chunks"] == [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
    assert seen["config"] == {"sample_rate": 16000, "channels": 1}
    assert result.text == "hello world"


def test_open_live_starts_the_request_before_any_audio(monkeypatch):
    # Given a session that has just been opened
    seen = _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()

        # Then the request is already in flight, which is the point: the
        # connection and config go out while the speaker is still talking
        assert seen["called"].wait(timeout=2.0)
        assert not seen["completed"]

        session.result()


def test_live_session_context_manager_ends_the_audio(monkeypatch):
    # Given audio written inside a with-block
    _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        with transcriber.open_live() as session:
            session.write(b"RIFF")
            assert not session.closed

        # Then leaving the block closes the audio and the result is waiting
        assert session.closed
        assert session.result().text == "hello world"


def test_live_session_rejects_writes_after_close(monkeypatch):
    # Given a closed session
    _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.close()

        # When more audio arrives, then it is refused rather than dropped silently
        with pytest.raises(RuntimeError, match="closed"):
            session.write(b"RIFF")

        session.result()


def test_live_session_rejects_text_at_write_time(monkeypatch):
    # Given a producer handing out str
    _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()

        # When it is written, then the mistake is named immediately, not when
        # the result is collected
        with pytest.raises(TypeError, match="bytes"):
            session.write("RIFF")

        session.result()


def test_live_session_abort_drops_the_request(monkeypatch):
    # Given a session mid-upload
    seen = _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")
        assert seen["called"].wait(timeout=2.0)

        # When it is aborted
        session.abort()

        # Then the transport never completed the upload, there is no result,
        # and a second abort is harmless
        assert not seen["completed"]
        assert session.closed
        with pytest.raises(RuntimeError, match="aborted"):
            session.result()
        session.abort()


def test_live_session_aborts_when_the_block_raises(monkeypatch):
    # Given a with-block that fails part-way through recording
    seen = _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        with pytest.raises(ValueError):
            with transcriber.open_live() as session:
                session.write(b"RIFF")
                raise ValueError("microphone unplugged")

        # Then the upload was dropped rather than transcribed
        assert not seen["completed"]
        with pytest.raises(RuntimeError, match="aborted"):
            session.result()


def test_live_session_abort_after_completion_is_a_no_op(monkeypatch):
    # Given a session whose result has already arrived
    _fake_live(monkeypatch, api, _drain)

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()
        result = session.result()

        # When it is aborted anyway, then nothing changes
        session.abort()
        assert session.result() is result


def test_live_session_rejects_job_api_config():
    # Given the job API's config type
    with aai.SyncTranscriber() as transcriber:
        with pytest.raises(TypeError, match="SyncTranscriptionConfig"):
            transcriber.open_live(aai.TranscriptionConfig())


def test_live_session_roundtrip_over_http(httpx_mock: HTTPXMock):
    # Given the real transport against a mocked endpoint
    _mock_ok(httpx_mock)

    with aai.SyncTranscriber() as transcriber:
        with transcriber.open_live() as session:
            session.write(b"RIFF")
            session.write(b"fake")

        # Then the request hit the live route and parsed like the pull path
        result = session.result()

    assert result.text == "hello world"
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == STREAM_URL
    assert request.headers.get("transfer-encoding") == "chunked"


def test_live_session_surfaces_server_errors(httpx_mock: HTTPXMock):
    # Given a rejection the server may send mid-upload
    httpx_mock.add_response(
        url=STREAM_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"status": 503, "title": "Capacity Exceeded", "detail": "no capacity"},
    )

    with aai.SyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")

        # When the result is collected, then it is the same error the pull path raises
        with pytest.raises(aai.SyncTranscriptError) as exc:
            session.result()

    assert exc.value.error_code == "capacity_exceeded"


@pytest.mark.asyncio
async def test_async_open_live_uploads_written_chunks_in_order(monkeypatch):
    # Given an asyncio session fed from a callback-style producer
    seen = _fake_live(monkeypatch, async_api, _adrain)

    async with aai.AsyncSyncTranscriber() as transcriber:
        session = transcriber.open_live(
            aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
        )
        for piece in (b"\x00\x01", bytearray(b"\x02\x03")):
            session.write(piece)

        # When the audio ends and the result is awaited
        result = await session.result()

    # Then it matches the sync twin
    assert seen["chunks"] == [b"\x00\x01", b"\x02\x03"]
    assert seen["config"] == {"sample_rate": 16000, "channels": 1}
    assert result.text == "hello world"


@pytest.mark.asyncio
async def test_async_live_session_context_manager_ends_the_audio(monkeypatch):
    # Given audio written inside an async with-block
    _fake_live(monkeypatch, async_api, _adrain)

    async with aai.AsyncSyncTranscriber() as transcriber:
        async with transcriber.open_live() as session:
            session.write(b"RIFF")

        # Then leaving the block closes the audio and the result is waiting
        assert session.closed
        assert (await session.result()).text == "hello world"


@pytest.mark.asyncio
async def test_async_live_session_rejects_writes_after_close(monkeypatch):
    # Given a closed session
    _fake_live(monkeypatch, async_api, _adrain)

    async with aai.AsyncSyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.close()

        # When more audio arrives, then it is refused
        with pytest.raises(RuntimeError, match="closed"):
            session.write(b"RIFF")

        await session.result()


@pytest.mark.asyncio
async def test_async_live_session_abort_drops_the_request(monkeypatch):
    # Given a session mid-upload
    seen = _fake_live(monkeypatch, async_api, _adrain)

    async with aai.AsyncSyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")
        await asyncio.sleep(0)  # let the task start consuming

        # When it is aborted
        await session.abort()

        # Then the transport never completed the upload and there is no result
        assert not seen["completed"]
        with pytest.raises(RuntimeError, match="aborted"):
            await session.result()
        await session.abort()


@pytest.mark.asyncio
async def test_async_live_session_aborts_when_the_block_raises(monkeypatch):
    # Given an async with-block that fails mid-recording
    seen = _fake_live(monkeypatch, async_api, _adrain)

    async with aai.AsyncSyncTranscriber() as transcriber:
        with pytest.raises(ValueError):
            async with transcriber.open_live() as session:
                session.write(b"RIFF")
                raise ValueError("call dropped")

        # Then the upload was dropped rather than transcribed
        assert not seen["completed"]
        with pytest.raises(RuntimeError, match="aborted"):
            await session.result()


@pytest.mark.asyncio
async def test_async_live_session_roundtrip_over_http(httpx_mock: HTTPXMock):
    # Given the real transport against a mocked endpoint
    _mock_ok(httpx_mock)

    async with aai.AsyncSyncTranscriber() as transcriber:
        async with transcriber.open_live() as session:
            session.write(b"RIFF")

        result = await session.result()

    # Then the request hit the live route and parsed like the pull path
    assert result.text == "hello world"
    assert str(httpx_mock.get_requests()[0].url) == STREAM_URL


@pytest.mark.asyncio
async def test_async_live_session_surfaces_server_errors(httpx_mock: HTTPXMock):
    # Given a rejection
    httpx_mock.add_response(
        url=STREAM_URL,
        method="POST",
        status_code=httpx.codes.TOO_MANY_REQUESTS,
        json={"status": 429, "title": "Rate Limited", "detail": "slow down"},
        headers={"Retry-After": "3"},
    )

    async with aai.AsyncSyncTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")

        # When the result is collected, then it is the same error the pull path raises
        with pytest.raises(aai.SyncTranscriptError) as exc:
            await session.result()

    assert exc.value.status_code == 429
    assert exc.value.retry_after == 3


def test_async_open_live_needs_a_running_loop():
    # Given no event loop
    transcriber = aai.AsyncSyncTranscriber()

    # When a session is opened, then the mistake is named rather than deferred
    with pytest.raises(RuntimeError, match="no running event loop"):
        transcriber.open_live()


def test_complete_audio_goes_over_the_live_connection(httpx_mock: HTTPXMock):
    """`transcribe()` opens the same connection `transcribe_live()` does.

    There is one request shape in this client: a complete clip is a stream
    whose bytes happen to all be ready, so it takes the streamed endpoint and
    the chunked framing rather than a second, buffered path.
    """
    httpx_mock.add_response(url=STREAM_URL, json=_OK_RESPONSE)

    aai.SyncTranscriber().transcribe(b"RIFFfake-wav-bytes")

    request = httpx_mock.get_requests()[0]
    assert str(request.url) == STREAM_URL
    # Chunked, not a declared length: the encoder frames an unsized iterator.
    assert "content-length" not in request.headers
    body = request.read()
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"RIFFfake-wav-bytes" in body


@pytest.mark.asyncio
async def test_async_complete_audio_goes_over_the_live_connection(
    httpx_mock: HTTPXMock,
):
    httpx_mock.add_response(url=STREAM_URL, json=_OK_RESPONSE)

    async with aai.AsyncSyncTranscriber() as transcriber:
        await transcriber.transcribe(b"RIFFfake-wav-bytes")

    request = httpx_mock.get_requests()[0]
    assert str(request.url) == STREAM_URL
    assert "content-length" not in request.headers


def test_the_buffered_endpoint_is_never_requested(httpx_mock: HTTPXMock):
    """No entry point on this client posts to `/v1/transcribe`.

    The path is still exported for the legacy `assemblyai.sync_api` surface,
    and the service still serves it — but nothing here sends to it, so a
    mocked buffered endpoint goes unused whichever way audio is submitted.
    """
    # One registration per call rather than a reusable one: the `is_reusable`
    # kwarg postdates the oldest pytest-httpx the matrix tests against.
    for _ in range(4):
        httpx_mock.add_response(url=STREAM_URL, json=_OK_RESPONSE)

    transcriber = aai.SyncTranscriber()
    transcriber.transcribe(b"RIFFfake-wav-bytes")
    transcriber.transcribe_async(b"RIFFfake-wav-bytes").result()
    transcriber.transcribe_live([b"RIFFfake", b"-wav-bytes"])
    with transcriber.open_live(aai.SyncTranscriptionConfig()) as session:
        session.write(b"RIFFfake-wav-bytes")
    session.result()

    requested = {str(request.url) for request in httpx_mock.get_requests()}
    assert requested == {STREAM_URL}
