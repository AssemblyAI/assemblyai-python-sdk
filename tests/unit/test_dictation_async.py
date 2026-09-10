"""Tests for the asyncio dictation client.

The framing itself is covered by `test_dictation.py`; these pin the asyncio
twin's behaviour: same route, same parsing, same errors, plus the async
producer shapes and the `AsyncDictationLiveSession`.
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
from assemblyai.dictation.v1 import async_api

pytestmark = pytest.mark.asyncio

aai.settings.api_key = "test"

LIVE_URL = f"{aai.settings.dictation_base_url}/v1/transcribe/live"
WARM_URL = f"{aai.settings.dictation_base_url}/v1/warm"

_OK_RESPONSE = {
    "text": "patient reports mild headache",
    "words": [
        {"text": "patient", "confidence": 0.9},
        {"text": "reports", "confidence": 0.95},
    ],
    "confidence": 0.92,
    "llm_response": None,
    "llm_error": None,
    "audio_duration_ms": 400,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 243.7,
    "sync_time_ms": 180.2,
}


def _mock_ok(httpx_mock: HTTPXMock, **overrides) -> None:
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={**_OK_RESPONSE, **overrides},
    )


def _chunks(*pieces: bytes):
    for piece in pieces:
        yield piece


async def _achunks(*pieces: bytes):
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


async def _asend(chunks, config: Optional[dict] = None) -> List[httpx.Request]:
    """Runs `async_api.transcribe_live` against a transport that records the body."""

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(httpx.codes.OK, json=_OK_RESPONSE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await async_api.transcribe_live(
            client,
            base_url=aai.settings.dictation_base_url,
            chunks=chunks,
            filename="audio.wav",
            audio_content_type="audio/wav",
            config=config,
            timeout=30.0,
        )

    return seen


class _SpyFile(io.BytesIO):
    """A file object that records the thread of every read asked of it."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.reads: List[int] = []
        self.threads: List[threading.Thread] = []

    def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        self.threads.append(threading.current_thread())
        return super().read(size)


# --- body shape -------------------------------------------------------------


async def test_body_matches_the_threaded_twin():
    # Given the same audio through the async transport
    # When the body is built
    request = (await _asend(_achunks(b"RIFF", b"fake", b"wav"), config={"a": 1}))[0]

    # Then it is framed identically: config first, chunked, no model header
    parts = dict(_parts(request))
    assert [name for name, _ in _parts(request)] == ["config", "audio"]
    assert parts["audio"].get_payload(decode=True) == b"RIFFfakewav"
    assert parts["config"].get_payload(decode=True) == b'{"a": 1}'
    assert request.headers.get("transfer-encoding") == "chunked"
    assert str(request.url) == LIVE_URL
    assert "X-AAI-Model" not in request.headers


async def test_body_sends_an_empty_config_part_when_there_is_no_config():
    request = (await _asend(_achunks(b"RIFF")))[0]

    parts = _parts(request)
    assert [name for name, _ in parts] == ["config", "audio"]
    assert dict(parts)["config"].get_payload(decode=True) == b"{}"


async def test_body_accepts_a_sync_iterable():
    # Given a plain iterable handed to the async transport
    request = (await _asend(_chunks(b"RIFF", b"fake")))[0]

    # Then one producer can feed both transports
    assert dict(_parts(request))["audio"].get_payload(decode=True) == b"RIFFfake"


async def test_body_reads_a_file_object_off_the_event_loop():
    # Given a file object handed to the async transport
    source = _SpyFile(b"RIFF" + bytes(_STREAM_READ_SIZE + 1))

    # When it is streamed
    request = (await _asend(source))[0]

    # Then every read ran in a worker thread, so a blocking read cannot stall
    # the loop, and the body is intact
    assert source.reads == [_STREAM_READ_SIZE] * 3
    assert all(t is not threading.main_thread() for t in source.threads)
    assert dict(_parts(request))["audio"].get_payload(decode=True) == source.getvalue()


# --- transcriber behaviour --------------------------------------------------


async def test_transcribe_live_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming from an async producer
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe_live(_achunks(b"RIFF", b"fake"))

    # Then the response is parsed into a DictationResponse
    assert isinstance(result, aai.DictationResponse)
    assert result.text == "patient reports mild headache"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[1].text == "reports"
    assert result.request_time_ms == 243.7
    assert result.sync_time_ms == 180.2


async def test_transcribe_live_posts_to_the_dictation_host(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe_live(_achunks(b"RIFF"))

    # Then the request goes to the dictation host with the raw key, no
    # routing header, and the dictation timeout
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert request.headers["authorization"] == "test"
    assert "X-AAI-Model" not in request.headers
    assert request.extensions["timeout"]["read"] == aai.settings.dictation_http_timeout


async def test_complete_audio_goes_over_the_live_connection(httpx_mock: HTTPXMock):
    # Given audio the caller already holds whole
    _mock_ok(httpx_mock)

    # When streaming it
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe_live(b"RIFFfake-wav-bytes")

    # Then it takes the live route as a single chunk, config first
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert "content-length" not in request.headers
    body = request.read()
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"RIFFfake-wav-bytes" in body


async def test_transcribe_live_reads_a_path_off_the_loop(
    httpx_mock: HTTPXMock, tmp_path
):
    # Given a local MP3 file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When streaming the path
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe_live(str(audio_file))

    # Then the file is sent whole with its true Content-Type and name
    body = httpx_mock.get_requests()[0].read()
    assert b"ID3fake-mp3-bytes" in body
    assert b"Content-Type: audio/mpeg" in body
    assert b'filename="note.mp3"' in body


async def test_transcribe_live_uses_default_config_and_per_call_override(
    httpx_mock: HTTPXMock,
):
    # Given a transcriber with a default config
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    default = aai.DictationConfig(llm_instruction="default instruction")

    async with aai.AsyncDictationTranscriber(config=default) as transcriber:
        # When streaming without a per-call config, and with an override
        await transcriber.transcribe_live(_achunks(b"RIFF"))
        override = aai.DictationConfig(llm_instruction="override instruction")
        await transcriber.transcribe_live(_achunks(b"RIFF"), config=override)

    # Then the default applies to the first call and the override to the second
    first, second = (request.read() for request in httpx_mock.get_requests())
    assert b"default instruction" in first
    assert b"override instruction" in second


async def test_transcribe_live_marks_pcm_from_config(httpx_mock: HTTPXMock):
    # Given a config carrying the fields only raw PCM needs
    _mock_ok(httpx_mock)
    config = aai.DictationConfig(sample_rate=16000, channels=1)

    # When streaming
    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe_live(_achunks(b"\x00\x01"), config=config)

    # Then the audio part selects the PCM decoder and the config carries both
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


async def test_transcribe_live_requires_both_pcm_fields():
    # Given a config with only half of what raw PCM needs
    config = aai.DictationConfig(sample_rate=16000)

    # When streaming, then it fails locally before any request
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(ValueError, match="sample_rate and channels"):
            await transcriber.transcribe_live(_achunks(b"\x00"), config=config)


async def test_transcribe_live_rejects_url():
    # Given an http URL as input
    async with aai.AsyncDictationTranscriber() as transcriber:
        # When streaming, then the message names this client, not its
        # threaded twin
        with pytest.raises(
            ValueError, match="AsyncDictationTranscriber does not accept URLs"
        ):
            await transcriber.transcribe_live("https://example.com/audio.wav")


async def test_transcribe_live_rejects_an_unsupported_source():
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(TypeError, match="unsupported audio source type"):
            await transcriber.transcribe_live(object())


async def test_transcribe_live_gather_runs_concurrently(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint answering twice
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)

    # When fanning two clips out with asyncio.gather on one transcriber
    async with aai.AsyncDictationTranscriber() as transcriber:
        results = await asyncio.gather(
            transcriber.transcribe_live(b"RIFFone"),
            transcriber.transcribe_live(_achunks(b"RIFFtwo")),
        )

    # Then both finish and parse
    assert [result.text for result in results] == [
        "patient reports mild headache",
        "patient reports mild headache",
    ]


async def test_final_text_prefers_llm_response(httpx_mock: HTTPXMock):
    # Given a response carrying an LLM rewrite
    _mock_ok(httpx_mock, llm_response="S: Mild headache.")

    # When streaming with an llm_instruction
    config = aai.DictationConfig(llm_instruction="Format this as a SOAP note.")
    async with aai.AsyncDictationTranscriber() as transcriber:
        result = await transcriber.transcribe_live(_achunks(b"RIFF"), config=config)

    # Then final_text is the rewrite and the raw transcript is still there
    assert result.final_text == "S: Mild headache."
    assert result.text == "patient reports mild headache"


async def test_error_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given a server rejecting the audio with an {error, error_code} body
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=415,
        json={"error": "unsupported media type", "error_code": "bad_audio"},
    )

    # When streaming, then a DictationError carries code + status
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe_live(_achunks(b"RIFF"))

    error = exc_info.value
    assert error.status_code == 415
    assert error.error_code == "bad_audio"
    assert not isinstance(error, aai.SyncTranscriptError)


async def test_rate_limit_surfaces_retry_after(httpx_mock: HTTPXMock):
    # Given a rate-limit response with a Retry-After header
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=429,
        json={"status": 429, "title": "Too Many Requests", "detail": "slow down"},
        headers={"Retry-After": "5"},
    )

    # When streaming, then retry_after and the snake_cased title are parsed
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(aai.DictationError) as exc_info:
            await transcriber.transcribe_live(_achunks(b"RIFF"))

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


async def test_warm_opens_connection(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    async with aai.AsyncDictationTranscriber() as transcriber:
        warmed = await transcriber.warm()

    # Then it returns True and probes the dictation warm route
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == WARM_URL
    assert request.method == "GET"
    assert "X-AAI-Model" not in request.headers


async def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    async with aai.AsyncDictationTranscriber() as transcriber:
        assert await transcriber.warm() is True


async def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    async with aai.AsyncDictationTranscriber() as transcriber:
        assert await transcriber.warm() is False


async def test_api_key_constructor_builds_own_client(httpx_mock: HTTPXMock):
    # Given a transcriber constructed with an explicit key
    _mock_ok(httpx_mock)

    # When streaming
    async with aai.AsyncDictationTranscriber(api_key="per-call-key") as transcriber:
        await transcriber.transcribe_live(_achunks(b"RIFF"))

    # Then that key authenticates the request
    assert httpx_mock.get_requests()[0].headers["authorization"] == "per-call-key"


async def test_rejects_sync_transcription_config():
    with pytest.raises(TypeError, match="expects DictationConfig"):
        aai.AsyncDictationTranscriber(config=aai.SyncTranscriptionConfig())


async def test_context_manager_closes_owned_client():
    # Given a transcriber that created its own client
    async with aai.AsyncDictationTranscriber() as transcriber:
        assert isinstance(transcriber, aai.AsyncDictationTranscriber)
        assert not transcriber.client.http_client.is_closed

    # Then leaving the block closes the owned connection pool
    assert transcriber.client.http_client.is_closed


async def test_aclose_leaves_shared_client_open():
    # Given a transcriber built on a caller-owned client
    async with aai.AsyncClient(settings=aai.settings) as client:
        transcriber = aai.AsyncDictationTranscriber(client=client)

        # When closing the transcriber
        await transcriber.aclose()

        # Then the shared pool stays open — its creator closes it
        assert not client.http_client.is_closed

    assert client.http_client.is_closed


# --- push-style sessions ------------------------------------------------------


def _fake_live(monkeypatch, error: Optional[Exception] = None) -> dict:
    """Replaces the async transport with a fake that drains the producer."""

    seen: dict = {"called": False, "completed": False}

    async def fake(client, **kwargs):
        seen["called"] = True
        seen["config"] = kwargs["config"]
        seen["chunks"] = [chunk async for chunk in kwargs["chunks"]]
        seen["completed"] = True
        if error:
            raise error
        return aai.DictationResponse.parse_obj(_OK_RESPONSE)

    monkeypatch.setattr(async_api, "transcribe_live", fake)
    return seen


async def test_open_live_uploads_written_chunks_in_order(monkeypatch):
    # Given an asyncio session fed from a callback-style producer
    seen = _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live(
            aai.DictationConfig(sample_rate=16000, channels=1)
        )
        for piece in (b"\x00\x01", bytearray(b"\x02\x03")):
            session.write(piece)

        # When the audio ends and the result is awaited
        result = await session.result()

    # Then it matches the threaded twin
    assert seen["chunks"] == [b"\x00\x01", b"\x02\x03"]
    assert seen["config"] == {"sample_rate": 16000, "channels": 1}
    assert result.text == "patient reports mild headache"


async def test_open_live_starts_the_request_before_any_audio(monkeypatch):
    # Given a session that has just been opened
    seen = _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()
        await asyncio.sleep(0)  # let the task start

        # Then the request is already in flight
        assert seen["called"]
        assert not seen["completed"]

        await session.result()


async def test_live_session_context_manager_ends_the_audio(monkeypatch):
    # Given audio written inside an async with-block
    _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        async with transcriber.open_live() as session:
            session.write(b"RIFF")
            assert not session.closed

        # Then leaving the block closes the audio and the result is waiting
        assert session.closed
        assert (await session.result()).text == "patient reports mild headache"


async def test_live_session_rejects_writes_after_close(monkeypatch):
    # Given a closed session
    _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()
        session.close()

        # When more audio arrives, then it is refused
        with pytest.raises(RuntimeError, match="closed"):
            session.write(b"RIFF")

        await session.result()


async def test_live_session_rejects_text_at_write_time(monkeypatch):
    _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()

        with pytest.raises(TypeError, match="bytes"):
            session.write("RIFF")

        await session.result()


async def test_live_session_abort_drops_the_request(monkeypatch):
    # Given a session mid-upload
    seen = _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")
        await asyncio.sleep(0)  # let the task start consuming

        # When it is aborted
        await session.abort()

        # Then the transport never completed the upload, there is no result,
        # and a second abort is harmless
        assert not seen["completed"]
        assert session.closed
        with pytest.raises(RuntimeError, match="aborted"):
            await session.result()
        await session.abort()


async def test_live_session_aborts_when_the_block_raises(monkeypatch):
    # Given an async with-block that fails mid-recording
    seen = _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(ValueError):
            async with transcriber.open_live() as session:
                session.write(b"RIFF")
                raise ValueError("call dropped")

        # Then the upload was dropped rather than transcribed
        assert not seen["completed"]
        with pytest.raises(RuntimeError, match="aborted"):
            await session.result()


async def test_live_session_abort_after_completion_is_a_no_op(monkeypatch):
    _fake_live(monkeypatch)

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()
        result = await session.result()

        await session.abort()
        assert await session.result() is result


async def test_live_session_rejects_other_products_configs():
    async with aai.AsyncDictationTranscriber() as transcriber:
        with pytest.raises(TypeError, match="expects DictationConfig"):
            transcriber.open_live(aai.SyncTranscriptionConfig())


async def test_live_session_roundtrip_over_http(httpx_mock: HTTPXMock):
    # Given the real transport against a mocked endpoint
    _mock_ok(httpx_mock)

    async with aai.AsyncDictationTranscriber() as transcriber:
        async with transcriber.open_live() as session:
            session.write(b"RIFF")

        result = await session.result()

    # Then the request hit the live route and parsed like the pull path
    assert result.text == "patient reports mild headache"
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert request.headers.get("transfer-encoding") == "chunked"


async def test_live_session_surfaces_server_errors(httpx_mock: HTTPXMock):
    # Given a rejection
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=httpx.codes.TOO_MANY_REQUESTS,
        json={"status": 429, "title": "Rate Limited", "detail": "slow down"},
        headers={"Retry-After": "3"},
    )

    async with aai.AsyncDictationTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")

        # When the result is collected, then it is the same error the pull path raises
        with pytest.raises(aai.DictationError) as exc:
            await session.result()

    assert exc.value.status_code == 429
    assert exc.value.retry_after == 3


@pytest.mark.asyncio(loop_scope="function")
async def test_the_buffered_endpoint_is_never_requested(httpx_mock: HTTPXMock):
    """No entry point on the asyncio client posts to `/v1/transcribe`."""
    httpx_mock.add_response(url=LIVE_URL, json=_OK_RESPONSE, is_reusable=True)

    async with aai.AsyncDictationTranscriber() as transcriber:
        await transcriber.transcribe_live(b"RIFFfake-wav-bytes")
        await transcriber.transcribe_live(_achunks(b"RIFFfake", b"-wav-bytes"))
        await transcriber.transcribe_live(io.BytesIO(b"RIFFfake-wav-bytes"))
        async with transcriber.open_live() as session:
            session.write(b"RIFFfake-wav-bytes")
        await session.result()

    requested = {str(request.url) for request in httpx_mock.get_requests()}
    assert requested == {LIVE_URL}


def test_open_live_needs_a_running_loop():
    # Given no event loop
    transcriber = aai.AsyncDictationTranscriber()

    # When a session is opened, then the mistake is named rather than deferred
    with pytest.raises(RuntimeError, match="no running event loop"):
        transcriber.open_live()
