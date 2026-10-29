"""Tests for the threaded dictation client.

Every entry point posts to the live route. Body-shape assertions go through
`httpx.MockTransport`, which consumes the request stream the way a real
transport does. `pytest_httpx` records the request without reading it on older
httpx, so a body read after the call sees a spent generator or a closed file.
"""

import email
import io
import json
import threading
import time
from typing import Iterable, List, Optional, cast

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai._multipart import _STREAM_READ_SIZE
from assemblyai.dictation.v1 import api
from assemblyai.dictation.v1.models import (
    _DICTATION_MAX_KEYTERMS_COUNT,
    _DICTATION_MAX_KEYTERMS_PROMPT_LEN,
    _DICTATION_MAX_STT_PROMPT_LEN,
)

aai.settings.api_key = "test"

LIVE_URL = f"{aai.settings.dictation_base_url}/v1/transcribe/live"
BUFFERED_URL = f"{aai.settings.dictation_base_url}/v1/transcribe"
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
            base_url=aai.settings.dictation_base_url,
            chunks=chunks,
            filename=kwargs.get("filename", "audio.wav"),
            audio_content_type=kwargs.get("audio_content_type", "audio/wav"),
            config=config,
            timeout=30.0,
        )

    return seen


# --- public surface ---------------------------------------------------------


def test_public_names_are_the_dictation_models_classes():
    # Given the module the dictation models live in
    from assemblyai.dictation.v1 import models

    # Then the top-level and versioned names are that module's classes
    assert aai.DictationConfig is models.DictationConfig
    assert aai.DictationError is models.DictationError
    assert aai.DictationResponse is models.DictationResponse
    assert aai.DictationWord is models.DictationWord


def test_there_is_no_buffered_transcribe():
    # Given the two clients
    # Then neither exposes a buffered entry point: every dictation starts on
    # the live route, and the buffered route is only its fallback past the
    # 120 s cap
    for cls in (aai.DictationTranscriber, aai.AsyncDictationTranscriber):
        assert not hasattr(cls, "transcribe")
        assert not hasattr(cls, "transcribe_async")
        assert hasattr(cls, "transcribe_live")
        assert hasattr(cls, "open_live")
    assert not hasattr(api, "transcribe")


def test_sessions_are_exported_from_the_dictation_packages():
    from assemblyai import dictation
    from assemblyai.dictation import v1

    for module in (dictation, v1):
        assert module.DictationLiveSession is v1.client.DictationLiveSession
        assert (
            module.AsyncDictationLiveSession
            is v1.async_client.AsyncDictationLiveSession
        )


# --- body shape -------------------------------------------------------------


def test_body_sends_config_before_audio():
    # Given a config alongside the audio
    # When the body is built
    request = _send(_chunks(b"RIFF", b"fake"), config={"language_codes": ["en"]})[0]

    # Then config precedes audio: the server decodes as bytes land and rejects
    # audio that arrives first
    assert [name for name, _ in _parts(request)] == ["config", "audio"]


def test_body_carries_config_json_and_joined_audio():
    # Given audio produced in several pieces
    # When the body is built
    request = _send(_chunks(b"RIFF", b"fake", b"wav"), config={"llm_instruction": "x"})[
        0
    ]

    # Then the pieces arrive as one audio part and the config part is JSON
    parts = dict(_parts(request))
    assert parts["audio"].get_payload(decode=True) == b"RIFFfakewav"
    assert parts["config"].get_payload(decode=True) == b'{"llm_instruction": "x"}'
    assert parts["config"].get_content_type() == "application/json"


def test_body_sends_an_empty_config_part_when_there_is_no_config():
    # Given no config
    # When the body is built
    request = _send(_chunks(b"RIFF"))[0]

    # Then a config part still precedes the audio, as an empty object: the live
    # endpoint rejects audio that no config part came before
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
    request = _send(source, filename="note.wav")[0]

    # Then it is read in fixed pieces until exhausted, never slurped whole
    assert source.reads == [_STREAM_READ_SIZE] * 4
    part = dict(_parts(request))["audio"]
    assert part.get_payload(decode=True) == payload
    assert part.get_filename() == "note.wav"


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


def test_body_hits_the_live_endpoint_without_a_model_header():
    # Given a streamed upload
    # When it is sent
    request = _send(_chunks(b"RIFF"))[0]

    # Then it lands on the live route, and the Dictation API picks its own
    # model, so no routing header is sent
    assert str(request.url) == LIVE_URL
    assert "X-AAI-Model" not in request.headers


# --- transcriber behaviour --------------------------------------------------


def test_transcribe_live_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming audio chunks
    result = aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF", b"fake"))

    # Then the response is parsed into a DictationResponse
    assert isinstance(result, aai.DictationResponse)
    assert result.text == "patient reports mild headache"
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert result.words[1].text == "reports"
    assert result.words[1].confidence == 0.95
    assert result.confidence == 0.92
    assert result.audio_duration_ms == 400
    assert result.request_time_ms == 243.7
    assert result.sync_time_ms == 180.2


def test_transcribe_live_sends_raw_api_key_to_the_dictation_host(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    # Then the request goes to the dictation host, not the sync host, with the
    # API key sent raw (no Bearer prefix) and no routing header
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert request.headers["authorization"] == "test"
    assert "X-AAI-Model" not in request.headers


def test_complete_audio_goes_over_the_live_connection(httpx_mock: HTTPXMock):
    """Bytes are a stream whose pieces are all ready at once.

    There is one request shape in this client: a complete clip takes the live
    endpoint and the chunked framing rather than a second, buffered path.
    """
    _mock_ok(httpx_mock)

    result = aai.DictationTranscriber().transcribe_live(b"RIFFfake-wav-bytes")

    assert result.text == "patient reports mild headache"
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert "content-length" not in request.headers
    body = request.read()
    assert body.index(b'name="config"') < body.index(b'name="audio"')
    assert b"RIFFfake-wav-bytes" in body
    assert b'filename="audio.wav"' in body
    assert b"Content-Type: audio/wav" in body


def test_transcribe_live_uses_the_dictation_timeout(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming audio
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    # Then the request gets the dictation budget, which must outlast the final
    # segment and the LLM pass
    timeout = httpx_mock.get_requests()[0].extensions["timeout"]
    assert timeout["read"] == aai.settings.dictation_http_timeout


def test_transcribe_live_uses_default_config_and_per_call_override(
    httpx_mock: HTTPXMock,
):
    # Given a transcriber with a default config
    _mock_ok(httpx_mock)
    _mock_ok(httpx_mock)
    default = aai.DictationConfig(llm_instruction="default instruction")
    transcriber = aai.DictationTranscriber(config=default)

    # When streaming without a per-call config, and with an override
    transcriber.transcribe_live(_chunks(b"RIFF"))
    override = aai.DictationConfig(llm_instruction="override instruction")
    transcriber.transcribe_live(_chunks(b"RIFF"), config=override)

    # Then the default applies to the first call and the override to the second
    first, second = (request.read() for request in httpx_mock.get_requests())
    assert b"default instruction" in first
    assert b"override instruction" in second


def test_config_json_excludes_unset_fields(httpx_mock: HTTPXMock):
    # Given a config that sets only llm_instruction
    _mock_ok(httpx_mock)
    config = aai.DictationConfig(llm_instruction="Format this as a SOAP note.")

    # When streaming
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"), config=config)

    # Then the config JSON carries that field and omits the unset ones
    body = httpx_mock.get_requests()[0].read()
    assert b"Format this as a SOAP note." in body
    assert b'"language_codes"' not in body
    assert b'"sample_rate"' not in body
    assert b'"keyterms_prompt"' not in body
    assert b'"rewrite_mode"' not in body


def test_transcribe_live_sends_keyterms_prompt_and_language_codes(
    httpx_mock: HTTPXMock,
):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming with keyterms and languages
    config = aai.DictationConfig(
        keyterms_prompt=["AssemblyAI", "  Universal  ", ""],
        language_codes=["en", "es"],
    )
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"), config=config)

    # Then the config part carries both, with keyterms normalized
    body = httpx_mock.get_requests()[0].read()
    assert b'"AssemblyAI"' in body
    assert b'"Universal"' in body  # whitespace stripped, empty term dropped
    assert b'"language_codes"' in body
    assert b'"es"' in body


def test_transcribe_live_marks_pcm_from_config(httpx_mock: HTTPXMock):
    # Given a config carrying the fields only raw PCM needs
    _mock_ok(httpx_mock)
    config = aai.DictationConfig(sample_rate=16000, channels=1)

    # When streaming
    aai.DictationTranscriber().transcribe_live(_chunks(b"\x00\x01"), config=config)

    # Then the audio part selects the PCM decoder and the config carries both
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/pcm" in body
    assert b'filename="audio.pcm"' in body
    assert b'"sample_rate"' in body
    assert b'"channels"' in body


def test_transcribe_live_requires_both_pcm_fields():
    # Given a config with only half of what raw PCM needs
    config = aai.DictationConfig(sample_rate=16000)

    # When streaming, then it is rejected before any request is made
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.DictationTranscriber().transcribe_live(_chunks(b"\x00"), config=config)


def test_transcribe_live_pcm_path_requires_rate_and_channels(tmp_path):
    # Given a .pcm file and a config with neither rate nor channels
    audio_file = tmp_path / "note.pcm"
    audio_file.write_bytes(b"\x00\x01" * 100)

    # When streaming it, then the extension alone routes it as PCM and the
    # missing fields are caught locally
    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.DictationTranscriber().transcribe_live(str(audio_file))


def test_transcribe_live_reads_a_path_and_maps_its_extension(
    httpx_mock: HTTPXMock, tmp_path
):
    # Given a local MP3 file
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.mp3"
    audio_file.write_bytes(b"ID3fake-mp3-bytes")

    # When streaming the path
    aai.DictationTranscriber().transcribe_live(str(audio_file))

    # Then the file is read and sent whole, with its true Content-Type and
    # name, so a format the live route does not decode is rejected by name
    body = httpx_mock.get_requests()[0].read()
    assert b"ID3fake-mp3-bytes" in body
    assert b"Content-Type: audio/mpeg" in body
    assert b'filename="note.mp3"' in body


def test_transcribe_live_accepts_a_pathlike(httpx_mock: HTTPXMock, tmp_path):
    # Given a local WAV file as a Path
    _mock_ok(httpx_mock)
    audio_file = tmp_path / "note.wav"
    audio_file.write_bytes(b"RIFFfake-wav-bytes")

    # When streaming the Path object
    result = aai.DictationTranscriber().transcribe_live(audio_file)

    # Then it succeeds and ships a WAV audio part under the file's name
    assert result.text == "patient reports mild headache"
    body = httpx_mock.get_requests()[0].read()
    assert b"Content-Type: audio/wav" in body
    assert b'filename="note.wav"' in body


def test_transcribe_live_names_a_file_object(httpx_mock: HTTPXMock):
    # Given a file object carrying a name
    _mock_ok(httpx_mock)
    stream = io.BytesIO(b"RIFF")
    stream.name = "/tmp/note.flac"

    # When it is streamed
    aai.DictationTranscriber().transcribe_live(stream)

    # Then the audio part takes its name and type from the file
    body = httpx_mock.get_requests()[0].read()
    assert b'filename="note.flac"' in body
    assert b"Content-Type: audio/flac" in body


def test_transcribe_live_rejects_url():
    # Given an http URL as input
    transcriber = aai.DictationTranscriber()

    # When streaming, then it is rejected with a pointer to Transcriber, and
    # the message names this client
    with pytest.raises(ValueError, match="DictationTranscriber does not accept URLs"):
        transcriber.transcribe_live("https://example.com/audio.wav")


def test_transcribe_live_rejects_an_async_iterable():
    # Given an async producer handed to the threaded transcriber
    async def chunks():
        yield b"RIFF"

    # When streaming it, then it is turned away before any request is made
    # and pointed at the transcriber that can drive it
    with pytest.raises(TypeError, match="AsyncDictationTranscriber"):
        aai.DictationTranscriber().transcribe_live(chunks())


def test_transcribe_live_rejects_an_unsupported_source():
    # Given something that is neither audio nor a producer of it
    with pytest.raises(TypeError, match="unsupported audio source type: int"):
        aai.DictationTranscriber().transcribe_live(42)


def test_transcribe_live_rejects_other_products_configs():
    # Given the sync API's and the job API's config types
    transcriber = aai.DictationTranscriber()

    # When streaming with either, then the mismatch is named
    with pytest.raises(TypeError, match="expects DictationConfig"):
        transcriber.transcribe_live(
            _chunks(b"RIFF"), config=aai.SyncTranscriptionConfig()
        )
    with pytest.raises(TypeError, match="expects DictationConfig"):
        transcriber.transcribe_live(_chunks(b"RIFF"), config=aai.TranscriptionConfig())


def test_final_text_prefers_llm_response(httpx_mock: HTTPXMock):
    # Given a response carrying an LLM rewrite
    _mock_ok(httpx_mock, llm_response="S: Mild headache.")

    # When streaming with an llm_instruction
    config = aai.DictationConfig(llm_instruction="Format this as a SOAP note.")
    result = aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"), config=config)

    # Then final_text is the rewrite and the raw transcript is still there
    assert result.final_text == "S: Mild headache."
    assert result.text == "patient reports mild headache"


def test_final_text_falls_back_to_text(httpx_mock: HTTPXMock):
    # Given a response whose LLM pass failed
    _mock_ok(httpx_mock, llm_error="llm timed out")

    # When streaming
    result = aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    # Then final_text falls back to the raw transcript
    assert result.final_text == "patient reports mild headache"
    assert result.llm_error == "llm timed out"


def test_response_ignores_unknown_fields(httpx_mock: HTTPXMock):
    # Given a server that added a field this SDK version predates
    _mock_ok(httpx_mock, auth_time_ms=1.4)

    # When streaming, then the extra key is ignored instead of failing
    result = aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))
    assert result.text == "patient reports mild headache"


def test_config_rejects_unknown_fields():
    # Given an option the Dictation API does not accept
    # When building the config, then it is rejected rather than dropped
    with pytest.raises(ValueError):
        aai.DictationConfig(word_boost=["AssemblyAI"])


def test_config_rejects_the_prompt_spelling():
    # Given the server-side alias for stt_prompt
    # When building the config, then the one spelling the SDK exposes wins —
    # sending both is a 400, so there is nothing to gain from accepting two
    with pytest.raises(ValueError):
        aai.DictationConfig(prompt="A doctor dictating a visit note.")


def test_transcribe_live_sends_stt_prompt(httpx_mock: HTTPXMock):
    # Given a mocked live endpoint
    _mock_ok(httpx_mock)

    # When streaming with transcription context
    config = aai.DictationConfig(stt_prompt="A doctor dictating a visit note.")
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"), config=config)

    # Then the config part carries it under the name the API documents
    body = httpx_mock.get_requests()[0].read()
    assert b'"stt_prompt"' in body
    assert b"A doctor dictating a visit note." in body


def test_stt_prompt_too_long_raises():
    # Given an stt_prompt exceeding the character cap
    with pytest.raises(ValueError):
        aai.DictationConfig(stt_prompt="x" * (_DICTATION_MAX_STT_PROMPT_LEN + 1))


def test_keyterms_prompt_too_long_raises():
    # Given a keyterms_prompt exceeding the character cap
    with pytest.raises(ValueError, match="characters"):
        aai.DictationConfig(
            keyterms_prompt=["x" * (_DICTATION_MAX_KEYTERMS_PROMPT_LEN + 1)]
        )


def test_keyterms_prompt_too_many_terms_raises():
    # Given more terms than the count cap allows, each short enough that the
    # character cap is not the binding constraint
    with pytest.raises(ValueError, match="terms"):
        aai.DictationConfig(keyterms_prompt=["t"] * (_DICTATION_MAX_KEYTERMS_COUNT + 1))


def test_keyterms_prompt_at_the_count_cap_accepted():
    # Given exactly as many terms as the cap allows
    config = aai.DictationConfig(keyterms_prompt=["t"] * _DICTATION_MAX_KEYTERMS_COUNT)

    # Then the cap is inclusive
    assert len(config.keyterms_prompt) == _DICTATION_MAX_KEYTERMS_COUNT


def test_llm_instruction_too_long_raises():
    # Given an llm_instruction exceeding the 2048-char cap
    with pytest.raises(ValueError):
        aai.DictationConfig(llm_instruction="x" * 3000)


def test_transcribe_live_sends_rewrite_mode(httpx_mock: HTTPXMock):
    # Given a config asking for the smart rewrite tier
    _mock_ok(httpx_mock)
    config = aai.DictationConfig(
        llm_instruction="Format this as a SOAP note.",
        rewrite_mode=aai.DictationRewriteMode.smart,
    )

    # When streaming
    aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"), config=config)

    # Then the config JSON carries the mode's wire value
    body = httpx_mock.get_requests()[0].read()
    assert b'"rewrite_mode": "smart"' in body


def test_rewrite_mode_accepts_the_wire_string():
    # Given rewrite_mode passed as a plain string
    config = aai.DictationConfig(rewrite_mode="fast")

    # Then it is parsed into the enum
    assert config.rewrite_mode is aai.DictationRewriteMode.fast


def test_rewrite_mode_rejects_an_unknown_tier():
    # Given a tier the API does not define
    with pytest.raises(ValueError):
        aai.DictationConfig(rewrite_mode="gemma")


def test_error_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given a server rejecting the audio with an {error, error_code} body
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=415,
        json={"error": "unsupported media type", "error_code": "bad_audio"},
    )

    # When streaming, then a DictationError carries code + status
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    error = exc_info.value
    assert error.status_code == 415
    assert error.error_code == "bad_audio"
    assert "unsupported media type" in str(error)
    # And it is not the sync API's error class
    assert not isinstance(error, aai.SyncTranscriptError)


def test_problem_details_envelope_maps_to_dictation_error(httpx_mock: HTTPXMock):
    # Given the server rejects oversized audio mid-upload with a problem-details body
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=413,
        json={"status": 413, "title": "Audio Too Large", "detail": "too long"},
    )

    # When streaming without the buffered fallback (which would retry this
    # one), then the snake_cased title becomes the error_code
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber(fallback_to_buffered=False).transcribe_live(
            _chunks(b"RIFF")
        )

    error = exc_info.value
    assert error.status_code == 413
    assert error.error_code == "audio_too_large"
    assert "too long" in str(error)


def test_detail_only_envelope_on_invalid_key(httpx_mock: HTTPXMock):
    # Given an invalid key, which the Dictation API answers 404
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=404,
        json={"detail": "Invalid API key"},
    )

    # When streaming, then the detail becomes the message and error_code
    # stays absent
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    error = exc_info.value
    assert error.status_code == 404
    assert error.error_code is None
    assert "Invalid API key" in str(error)


def test_rate_limit_surfaces_retry_after(httpx_mock: HTTPXMock):
    # Given a rate-limit response with a Retry-After header
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=429,
        json={
            "status": 429,
            "title": "Too Many Requests",
            "detail": "Too many requests",
        },
        headers={"Retry-After": "5"},
    )

    # When streaming, then retry_after and the snake_cased title are parsed
    with pytest.raises(aai.DictationError) as exc_info:
        aai.DictationTranscriber().transcribe_live(_chunks(b"RIFF"))

    error = exc_info.value
    assert error.status_code == 429
    assert error.error_code == "too_many_requests"
    assert error.retry_after == 5


def test_warm_opens_connection(httpx_mock: HTTPXMock):
    # Given a mocked warm endpoint
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=httpx.codes.OK)

    # When warming the transcriber
    warmed = aai.DictationTranscriber().warm()

    # Then it returns True and probes the dictation warm route
    assert warmed is True
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == WARM_URL
    assert request.method == "GET"
    assert "X-AAI-Model" not in request.headers


def test_warm_returns_true_on_non_200(httpx_mock: HTTPXMock):
    # Given a warm route that the load balancer answers with a 404
    httpx_mock.add_response(url=WARM_URL, method="GET", status_code=404)

    # When warming, then the socket is still established, so warm() is True
    assert aai.DictationTranscriber().warm() is True


def test_warm_returns_false_on_transport_error(httpx_mock: HTTPXMock):
    # Given the dictation host is unreachable
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    # When warming, then the failure is swallowed and reported as False
    assert aai.DictationTranscriber().warm() is False


def test_api_key_constructor_builds_own_client(httpx_mock: HTTPXMock):
    # Given a transcriber constructed with an explicit key
    _mock_ok(httpx_mock)

    # When streaming
    aai.DictationTranscriber(api_key="per-call-key").transcribe_live(_chunks(b"RIFF"))

    # Then that key authenticates the request
    assert httpx_mock.get_requests()[0].headers["authorization"] == "per-call-key"


def test_rejects_sync_transcription_config():
    # Given the sync API's config, a different non-interchangeable type
    # When constructing, then the mistake is named
    with pytest.raises(TypeError, match="expects DictationConfig"):
        aai.DictationTranscriber(config=aai.SyncTranscriptionConfig())


def test_config_setter_validates_and_round_trips():
    # Given a transcriber and a new config
    transcriber = aai.DictationTranscriber()
    config = aai.DictationConfig(language_codes=["de"])

    # When assigning it
    transcriber.config = config

    # Then it is the transcriber's default
    assert transcriber.config is config

    # And a wrong type is rejected
    with pytest.raises(TypeError, match="expects DictationConfig"):
        transcriber.config = aai.SyncTranscriptionConfig()


def test_context_manager_returns_self_and_closes():
    # Given a transcriber used as a context manager
    with aai.DictationTranscriber() as transcriber:
        # Then the bound value is the transcriber itself
        assert isinstance(transcriber, aai.DictationTranscriber)

    # And leaving the block shuts the worker pool down
    assert transcriber._executor._shutdown is True


def test_dictation_settings_defaults():
    # Given default settings
    settings = aai.Settings()

    # Then the dictation host and the 300s per-operation timeout are in place
    assert settings.dictation_base_url == "https://dictation.assemblyai.com"
    assert settings.dictation_http_timeout == 300.0


def test_the_buffered_endpoint_is_not_requested_within_the_cap(
    httpx_mock: HTTPXMock,
):
    """No entry point posts to `/v1/transcribe` while the live request succeeds.

    The buffered route is only the fallback for a live 413 past the 120 s cap;
    whichever way audio is submitted, a dictation within it goes live only.
    """
    for _ in range(4):  # one per request; older pytest-httpx has no is_reusable
        _mock_ok(httpx_mock)

    with aai.DictationTranscriber() as transcriber:
        transcriber.transcribe_live(b"RIFFfake-wav-bytes")
        transcriber.transcribe_live([b"RIFFfake", b"-wav-bytes"])
        transcriber.transcribe_live(io.BytesIO(b"RIFFfake-wav-bytes"))
        with transcriber.open_live(aai.DictationConfig()) as session:
            session.write(b"RIFFfake-wav-bytes")
        session.result()

    requested = {str(request.url) for request in httpx_mock.get_requests()}
    assert requested == {LIVE_URL}


# --- push-style sessions ------------------------------------------------------


def _fake_live(monkeypatch, error: Optional[Exception] = None) -> dict:
    """
    Replaces the transport with a fake that drains the producer the way a real
    transport would, recording what it saw. An abort raised by the producer
    propagates from `list()` exactly as it would from httpx.
    """
    seen: dict = {"called": threading.Event(), "completed": False}

    def fake(client, **kwargs):
        seen["called"].set()
        seen["config"] = kwargs["config"]
        seen["chunks"] = list(kwargs["chunks"])
        seen["completed"] = True
        if error:
            raise error
        return aai.DictationResponse.parse_obj(_OK_RESPONSE)

    monkeypatch.setattr(api, "transcribe_live", fake)
    return seen


def test_open_live_uploads_written_chunks_in_order(monkeypatch):
    # Given a session fed from a callback-style producer
    seen = _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live(
            aai.DictationConfig(sample_rate=16000, channels=1)
        )
        for piece in (b"\x00\x01", bytearray(b"\x02\x03"), memoryview(b"\x04\x05")):
            session.write(piece)

        # When the audio ends and the result is awaited
        result = session.result()

    # Then the chunks went out in order, as bytes, with the config, and the
    # transcript came back parsed
    assert seen["chunks"] == [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
    assert seen["config"] == {"sample_rate": 16000, "channels": 1}
    assert result.text == "patient reports mild headache"


def test_open_live_starts_the_request_before_any_audio(monkeypatch):
    # Given a session that has just been opened
    seen = _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live()

        # Then the request is already in flight, which is the point: the
        # connection and config go out while the speaker is still talking
        assert seen["called"].wait(timeout=2.0)
        assert not seen["completed"]

        session.result()


def test_open_live_uses_the_transcribers_default_config(monkeypatch):
    # Given a transcriber with a default config and a session opened without one
    seen = _fake_live(monkeypatch)

    with aai.DictationTranscriber(
        config=aai.DictationConfig(llm_instruction="Summarize.")
    ) as transcriber:
        session = transcriber.open_live()
        session.result()

    # Then the default config rode along
    assert seen["config"] == {"llm_instruction": "Summarize."}


def test_live_session_context_manager_ends_the_audio(monkeypatch):
    # Given audio written inside a with-block
    _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        with transcriber.open_live() as session:
            session.write(b"RIFF")
            assert not session.closed

        # Then leaving the block closes the audio and the result is waiting
        assert session.closed
        assert session.result().text == "patient reports mild headache"


def test_live_session_rejects_writes_after_close(monkeypatch):
    # Given a closed session
    _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live()
        session.close()

        # When more audio arrives, then it is refused rather than dropped silently
        with pytest.raises(RuntimeError, match="closed"):
            session.write(b"RIFF")

        session.result()


def test_live_session_rejects_text_at_write_time(monkeypatch):
    # Given a producer handing out str
    _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live()

        # When it is written, then the mistake is named immediately, not when
        # the result is collected
        with pytest.raises(TypeError, match="bytes"):
            session.write("RIFF")

        session.result()


def test_live_session_abort_drops_the_request(monkeypatch):
    # Given a session mid-upload
    seen = _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
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
    seen = _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
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
    _fake_live(monkeypatch)

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live()
        result = session.result()

        # When it is aborted anyway, then nothing changes
        session.abort()
        assert session.result() is result


def test_live_session_rejects_other_products_configs():
    # Given the sync API's config type
    with aai.DictationTranscriber() as transcriber:
        with pytest.raises(TypeError, match="expects DictationConfig"):
            transcriber.open_live(aai.SyncTranscriptionConfig())


def test_live_session_roundtrip_over_http(httpx_mock: HTTPXMock):
    # Given the real transport against a mocked endpoint
    _mock_ok(httpx_mock)

    with aai.DictationTranscriber() as transcriber:
        with transcriber.open_live() as session:
            session.write(b"RIFF")
            session.write(b"fake")

        # Then the request hit the live route and parsed like the pull path
        result = session.result()

    assert result.text == "patient reports mild headache"
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == LIVE_URL
    assert request.headers.get("transfer-encoding") == "chunked"


def test_live_session_surfaces_server_errors(httpx_mock: HTTPXMock):
    # Given a rejection the server may send mid-upload
    httpx_mock.add_response(
        url=LIVE_URL,
        method="POST",
        status_code=httpx.codes.SERVICE_UNAVAILABLE,
        json={"status": 503, "title": "Capacity Exceeded", "detail": "no capacity"},
    )

    with aai.DictationTranscriber() as transcriber:
        session = transcriber.open_live()
        session.write(b"RIFF")

        # When the result is collected, then it is the same error the pull path raises
        with pytest.raises(aai.DictationError) as exc:
            session.result()

    assert exc.value.error_code == "capacity_exceeded"


# --- buffered fallback past the live 120 s cap --------------------------------

# The live endpoint's answer to audio past its cap, verbatim from production.
_LIVE_CAP = {"error": "audio exceeds 120000 ms", "error_code": "audio_too_large"}

_BUFFERED_OK = {
    **_OK_RESPONSE,
    "audio_duration_ms": 130000,
    "session_id": "4f0c1d7e-buffered",
}

_PCM = aai.DictationConfig(sample_rate=16000, channels=1)


class _CapTransport(httpx.BaseTransport):
    """
    A transport scripted like the Dictation API around the live 120 s cap.

    The live request's body is read for `live_pieces` pieces — the multipart
    head is the first — or whole when it is None, and then answered with
    `live_status`/`live_json`. Stopping part-way is what httpcore does when
    the server answers while the upload is still in flight: it stops pulling
    the body and returns the response. The buffered request is read whole and
    answered with `buffered_status`/`buffered_json`.
    """

    def __init__(
        self,
        *,
        live_status: int = 413,
        live_json: Optional[dict] = None,
        live_pieces: Optional[int] = None,
        buffered_status: int = 200,
        buffered_json: Optional[dict] = None,
        gate: Optional[threading.Event] = None,
    ) -> None:
        self.live_status = live_status
        self.live_json = _LIVE_CAP if live_json is None else live_json
        self.live_pieces = live_pieces
        self.buffered_status = buffered_status
        self.buffered_json = _BUFFERED_OK if buffered_json is None else buffered_json
        self.gate = gate
        self.live_request: Optional[httpx.Request] = None
        self.live_body = b""
        self.body_read = threading.Event()
        self.answered = threading.Event()
        self.buffered: List[tuple] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/transcribe/live":
            self.live_request = request
            pieces: List[bytes] = []
            for piece in cast(Iterable[bytes], request.stream):
                pieces.append(piece)
                if self.live_pieces is not None and len(pieces) >= self.live_pieces:
                    break
            self.live_body = b"".join(pieces)
            self.body_read.set()
            if self.gate is not None:
                self.gate.wait(timeout=5.0)
            self.answered.set()
            return httpx.Response(self.live_status, json=self.live_json)

        body = b"".join(cast(Iterable[bytes], request.stream))
        self.buffered.append((request, body))
        return httpx.Response(self.buffered_status, json=self.buffered_json)


def _transcriber_over(transport: httpx.BaseTransport, **kwargs):
    """A `DictationTranscriber` whose client sends through `transport`."""

    client = aai.Client()
    client.http_client.close()
    client._http_client = httpx.Client(
        transport=transport, headers=dict(client.http_client.headers)
    )
    return aai.DictationTranscriber(client=client, **kwargs)


def _body_parts(content_type: str, body: bytes):
    raw = b"Content-Type: " + content_type.encode() + b"\r\n\r\n"
    message = email.message_from_bytes(raw + body)

    return {
        part.get_param("name", header="content-disposition"): part
        for part in message.get_payload()
    }


def _audio_head(body: bytes) -> bytes:
    """The audio part's headers as the live body carried them."""

    start = body.index(b'name="audio"')
    return body[start : body.index(b"\r\n\r\n", start)]


def _assert_buffered(transport: _CapTransport, audio: bytes, config: dict) -> None:
    """The fallback posted `audio` whole, framed exactly like the live upload."""

    assert len(transport.buffered) == 1
    request, body = transport.buffered[0]
    assert str(request.url) == BUFFERED_URL
    assert request.headers["content-length"] == str(len(body))
    assert "transfer-encoding" not in request.headers
    assert request.headers["authorization"] == "test"

    parts = _body_parts(request.headers["content-type"], body)
    assert list(parts) == ["config", "audio"]
    assert json.loads(parts["config"].get_payload(decode=True)) == config
    assert parts["audio"].get_payload(decode=True) == audio
    # Same filename and Content-Type on the audio part as the live upload
    assert _audio_head(body) == _audio_head(transport.live_body)


def test_fallback_is_on_by_default_and_can_be_turned_off():
    assert aai.DictationTranscriber().fallback_to_buffered is True
    assert (
        aai.DictationTranscriber(fallback_to_buffered=False).fallback_to_buffered
        is False
    )


@pytest.mark.parametrize("fallback", [True, False])
def test_fallback_leaves_the_success_path_unchanged(fallback):
    # Given a dictation within the cap, which the live endpoint answers 200
    transport = _CapTransport(live_status=200, live_json=_OK_RESPONSE)

    # When it is sent with the fallback on or off
    result = _transcriber_over(
        transport, fallback_to_buffered=fallback
    ).transcribe_live(_chunks(b"\x01\x02", b"\x03\x04"), config=_PCM)

    # Then it is the live result, from the live route only, with the same
    # chunked body either way
    assert result.session_id == _OK_RESPONSE["session_id"]
    assert transport.buffered == []
    assert transport.live_request.headers["transfer-encoding"] == "chunked"
    parts = _body_parts(
        transport.live_request.headers["content-type"], transport.live_body
    )
    assert parts["audio"].get_payload(decode=True) == b"\x01\x02\x03\x04"


def test_live_cap_mid_upload_falls_back_with_all_the_audio():
    # Given the live endpoint answering 413 audio_too_large after the head and
    # the first chunk, while the producer is still recording
    transport = _CapTransport(live_pieces=2)
    pulled = []

    def producer():
        for chunk in (b"A" * 4, b"B" * 4, b"C" * 4, b"D" * 4):
            pulled.append((chunk[:1], transport.answered.is_set()))
            yield chunk

    # When the dictation runs
    result = _transcriber_over(transport).transcribe_live(producer(), config=_PCM)

    # Then the producer went on being read after the 413 until it ended
    assert pulled == [(b"A", False), (b"B", True), (b"C", True), (b"D", True)]
    # And the whole recording went to the buffered endpoint once, with the
    # same config, and its result came back like a live one
    _assert_buffered(
        transport,
        b"A" * 4 + b"B" * 4 + b"C" * 4 + b"D" * 4,
        {"sample_rate": 16000, "channels": 1},
    )
    assert isinstance(result, aai.DictationResponse)
    assert result.session_id == _BUFFERED_OK["session_id"]
    assert result.audio_duration_ms == 130000


def test_live_cap_after_the_whole_body_falls_back():
    # Given a 413 that arrives only once the whole body was sent — what
    # production does over HTTP/1.1, where it reads the body to the end first
    transport = _CapTransport()

    # When the dictation runs
    result = _transcriber_over(transport).transcribe_live(
        _chunks(b"RIFF", b"wav-1", b"wav-2"),
        config=aai.DictationConfig(language_codes=["en"]),
    )

    # Then the buffered request carries the same audio and config
    _assert_buffered(transport, b"RIFFwav-1wav-2", {"language_codes": ["en"]})
    assert result.session_id == _BUFFERED_OK["session_id"]


def test_live_cap_drains_the_rest_of_a_file_object():
    # Given a file object the live upload had read only part of
    audio = bytes(range(256)) * ((3 * _STREAM_READ_SIZE) // 256 + 1)
    transport = _CapTransport(live_pieces=2)

    # When the dictation runs
    _transcriber_over(transport).transcribe_live(io.BytesIO(audio), config=_PCM)

    # Then the rest of the file was read into the buffered request
    _assert_buffered(transport, audio, {"sample_rate": 16000, "channels": 1})


def test_fallback_disabled_raises_the_live_413_as_before():
    # Given a transcriber built without the fallback
    transport = _CapTransport(live_pieces=2)
    pulled = []

    def producer():
        for chunk in (b"A", b"B", b"C"):
            pulled.append(chunk)
            yield chunk

    # When the live endpoint answers 413 mid-upload
    with pytest.raises(aai.DictationError) as exc:
        _transcriber_over(transport, fallback_to_buffered=False).transcribe_live(
            producer(), config=_PCM
        )

    # Then the live error is raised, the producer is not read further and
    # nothing is retried
    assert exc.value.status_code == 413
    assert exc.value.error_code == "audio_too_large"
    assert "120000 ms" in str(exc.value)
    assert pulled == [b"A"]
    assert transport.buffered == []


@pytest.mark.parametrize(
    "status, body",
    [
        (413, {"error": "body too large", "error_code": "payload_too_large"}),
        (413, {"status": 413, "title": "Request Entity Too Large"}),
        (413, {"detail": "too large"}),
        (401, {"detail": "Invalid API key"}),
        (429, {"status": 429, "title": "Too Many Requests", "detail": "slow down"}),
        (500, {"error": "boom", "error_code": "audio_too_large"}),
        (503, {"status": 503, "title": "Capacity Exceeded"}),
    ],
)
def test_other_failures_do_not_fall_back(status, body):
    # Given a live failure other than 413 audio_too_large
    transport = _CapTransport(live_status=status, live_json=body, live_pieces=2)

    # When the dictation runs, then that error is raised as before and
    # nothing is retried
    with pytest.raises(aai.DictationError) as exc:
        _transcriber_over(transport).transcribe_live(_chunks(b"A", b"B"), config=_PCM)

    assert exc.value.status_code == status
    assert transport.buffered == []


@pytest.mark.parametrize(
    "status, body, code",
    [
        (
            413,
            {"error": "audio exceeds 1200000 ms", "error_code": "audio_too_large"},
            "audio_too_large",
        ),
        (500, {"status": 500, "title": "Internal Error"}, "internal_error"),
    ],
)
def test_a_failed_fallback_raises_its_own_error(status, body, code):
    # Given a buffered request that fails too
    transport = _CapTransport(buffered_status=status, buffered_json=body)

    # When the dictation runs
    with pytest.raises(aai.DictationError) as exc:
        _transcriber_over(transport).transcribe_live(_chunks(b"A"), config=_PCM)

    # Then the buffered request's error is raised, not the live 413
    assert len(transport.buffered) == 1
    assert exc.value.status_code == status
    assert exc.value.error_code == code
    assert "120000 ms" not in str(exc.value)


def test_the_copy_is_capped_while_draining(monkeypatch):
    # Given a copy cap of 10 bytes and a producer that keeps recording
    monkeypatch.setattr(api, "BUFFERED_MAX_BYTES", 10)
    transport = _CapTransport(live_pieces=2)
    pulled = []

    def producer():
        for _ in range(100):
            pulled.append(1)
            yield b"x" * 4

    # When the live request hits the cap and the producer is drained
    with pytest.raises(aai.DictationError) as exc:
        _transcriber_over(transport).transcribe_live(producer(), config=_PCM)

    # Then draining stops once the copy passes the cap, with a 413
    # audio_too_large, and nothing is sent
    assert exc.value.status_code == 413
    assert exc.value.error_code == "audio_too_large"
    assert len(pulled) == 3
    assert transport.buffered == []


def test_buffered_max_bytes_matches_the_buffered_endpoint():
    assert api.BUFFERED_MAX_BYTES == 256 * 1024 * 1024
    assert api.ENDPOINT_TRANSCRIBE == "/v1/transcribe"


def test_a_producer_error_while_draining_propagates_without_fallback():
    # Given a producer that fails after the live 413
    transport = _CapTransport(live_pieces=2)

    def producer():
        yield b"A"
        raise ValueError("microphone unplugged")

    # When the dictation runs, then the producer's error propagates unchanged
    # and nothing is sent
    with pytest.raises(ValueError, match="microphone unplugged"):
        _transcriber_over(transport).transcribe_live(producer(), config=_PCM)

    assert transport.buffered == []


def test_open_live_falls_back_with_audio_written_after_the_413():
    # Given a session whose live request is answered 413 after the first chunk
    transport = _CapTransport(live_pieces=2)

    with _transcriber_over(transport) as transcriber:
        session = transcriber.open_live(_PCM)
        session.write(b"A" * 4)
        assert transport.answered.wait(timeout=5.0)

        # When the speaker keeps talking, then stops
        session.write(b"B" * 4)
        session.write(b"C" * 4)
        session.close()
        result = session.result()

    # Then the buffered request carries everything written, and its result is
    # the session's
    _assert_buffered(
        transport, b"A" * 4 + b"B" * 4 + b"C" * 4, {"sample_rate": 16000, "channels": 1}
    )
    assert result.session_id == _BUFFERED_OK["session_id"]


def test_open_live_abort_after_the_413_sends_no_fallback():
    # Given a session whose live request was answered 413 mid-recording
    transport = _CapTransport(live_pieces=2)

    with _transcriber_over(transport) as transcriber:
        session = transcriber.open_live(_PCM)
        session.write(b"A")
        assert transport.answered.wait(timeout=5.0)
        session.write(b"B")

        # When it is aborted while the producer is being drained
        session.abort()

        # Then no buffered request is made and there is no result
        assert transport.buffered == []
        with pytest.raises(RuntimeError, match="aborted"):
            session.result()


def test_open_live_abort_after_close_sends_no_fallback():
    # Given a closed session whose live 413 has not arrived yet
    gate = threading.Event()
    transport = _CapTransport(gate=gate)

    with _transcriber_over(transport) as transcriber:
        session = transcriber.open_live(_PCM)
        session.write(b"A")
        session.close()
        assert transport.body_read.wait(timeout=5.0)

        # When it is aborted before the 413 comes back
        aborter = threading.Thread(target=session.abort)
        aborter.start()
        while not session._aborted:
            time.sleep(0.001)
        gate.set()
        aborter.join(timeout=5.0)

        # Then the fallback is not sent
        assert transport.answered.is_set()
        assert transport.buffered == []
        with pytest.raises(RuntimeError, match="aborted"):
            session.result()


def test_the_copy_is_capped_during_the_live_upload_too(monkeypatch):
    # Given a copy cap of 10 bytes, and a live endpoint that reads the whole
    # body before answering — as production does
    monkeypatch.setattr(api, "BUFFERED_MAX_BYTES", 10)
    transport = _CapTransport()
    pulled = []

    def producer():
        for _ in range(100):
            pulled.append(1)
            yield b"x" * 4

    # When the copy passes the cap while the live request is still uploading
    with pytest.raises(aai.DictationError) as exc:
        _transcriber_over(transport).transcribe_live(producer(), config=_PCM)

    # Then the upload stops there with a 413 audio_too_large; memory stays
    # bounded rather than growing until the speaker stops
    assert exc.value.status_code == 413
    assert exc.value.error_code == "audio_too_large"
    assert len(pulled) == 3
    assert transport.buffered == []


def test_the_copy_cap_does_not_apply_without_the_fallback(monkeypatch):
    # Given no fallback, so no copy is kept
    monkeypatch.setattr(api, "BUFFERED_MAX_BYTES", 10)
    transport = _CapTransport(live_status=200, live_json=_OK_RESPONSE)

    # When more than the cap is uploaded, then it goes out as before
    _transcriber_over(transport, fallback_to_buffered=False).transcribe_live(
        _chunks(*([b"x" * 4] * 10)), config=_PCM
    )
    assert transport.live_body.count(b"x") == 40
