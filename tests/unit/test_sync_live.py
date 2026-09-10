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
from assemblyai.sync.v1 import api, async_api
from assemblyai.sync.v1._multipart import _STREAM_READ_SIZE

aai.settings.api_key = "test"

STREAM_URL = f"{aai.settings.sync_base_url}/v1/transcribe/stream"

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
