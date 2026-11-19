"""Tests for the Diarization API client.

Every call posts one buffered multipart body (`config`, `transcript`,
`audio`) to `/v1/diarize` on the diarization host and parses the JSON result.
"""

import email
import io
import json
from typing import Any, Dict, List

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai

aai.settings.api_key = "test"

DIARIZE_URL = f"{aai.settings.diarization_base_url}/v1/diarize"

_WORDS: List[Dict[str, Any]] = [
    {"text": "hello", "start": 0, "end": 400, "confidence": 0.9},
    {"text": "hi", "start": 4400, "end": 4700, "confidence": 0.95},
]

_OK_RESPONSE = {
    "words": [
        {**_WORDS[0], "speaker": "A", "speaker_confidence": 0.93},
        {**_WORDS[1], "speaker": "B"},
    ],
    "utterances": [
        {
            "speaker": "A",
            "start": 0,
            "end": 400,
            "text": "hello",
            "confidence": 0.9,
            "speaker_confidence": 0.93,
            "words": [{**_WORDS[0], "speaker": "A", "speaker_confidence": 0.93}],
        },
        {
            "speaker": "B",
            "start": 4400,
            "end": 4700,
            "text": "hi",
            "confidence": 0.95,
            "words": [{**_WORDS[1], "speaker": "B"}],
        },
    ],
    "segments": [
        {"start": 0, "end": 4200, "speaker": "A", "speaker_confidence": 0.93},
        {"start": 4350, "end": 9100, "speaker": "B"},
    ],
    "speakers": ["A", "B"],
    "audio_duration_ms": 9500,
    "session_id": "eb92c4ff-4bbb-429f-9b99-7279d7fe738f",
    "request_time_ms": 312.4,
}

_SEGMENTS_ONLY = {**_OK_RESPONSE, "words": [], "utterances": []}


def _mock_ok(httpx_mock: HTTPXMock, body: dict = _OK_RESPONSE, url: str = DIARIZE_URL):
    httpx_mock.add_response(
        url=url, method="POST", status_code=httpx.codes.OK, json=body
    )


def _parts(request: httpx.Request) -> dict:
    """The recorded multipart body as {name: email part}, in order."""

    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n"
    message = email.message_from_bytes(raw + request.read())
    return {
        part.get_param("name", header="content-disposition"): part
        for part in message.get_payload()
    }


def _json_part(part) -> dict:
    assert part.get_content_type() == "application/json"
    return json.loads(part.get_payload(decode=True))


def _sent(httpx_mock: HTTPXMock) -> dict:
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    return _parts(requests[0])


def test_diarize_with_a_transcript(httpx_mock: HTTPXMock):
    # GIVEN a diarization host that labels the words
    _mock_ok(httpx_mock)

    # WHEN WAV audio is diarized with a word list
    result = aai.Diarizer().diarize(b"RIFF....", transcript={"words": _WORDS})

    # THEN one buffered request carried the raw key, the transcript and audio
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == DIARIZE_URL
    assert request.headers["authorization"] == "test"
    assert "content-length" in request.headers
    parts = _sent(httpx_mock)
    assert list(parts) == ["transcript", "audio"]
    assert _json_part(parts["transcript"]) == {"words": _WORDS}
    assert parts["audio"].get_content_type() == "audio/wav"
    assert parts["audio"].get_payload(decode=True) == b"RIFF...."

    # AND the result is parsed
    assert isinstance(result, aai.DiarizationResponse)
    assert [w.speaker for w in result.words] == ["A", "B"]
    assert result.words[0].speaker_confidence == 0.93
    assert result.words[1].speaker_confidence is None
    assert result.utterances[1].text == "hi"
    assert result.utterances[0].words[0].text == "hello"
    assert result.segments[1].speaker_confidence is None
    assert result.speakers == ["A", "B"]
    assert result.audio_duration_ms == 9500
    assert result.session_id == _OK_RESPONSE["session_id"]


def test_diarize_without_a_transcript_runs_segments_only(httpx_mock: HTTPXMock):
    # GIVEN a host answering segments only
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)

    # WHEN audio is diarized alone
    result = aai.Diarizer().diarize(b"RIFF....")

    # THEN neither a transcript nor a config part was sent
    assert list(_sent(httpx_mock)) == ["audio"]

    # AND the segments and speakers come back with no words
    assert result.words == [] and result.utterances == []
    assert [s.speaker for s in result.segments] == ["A", "B"]
    assert result.speakers == ["A", "B"]


def test_diarize_sends_an_explicit_empty_word_list(httpx_mock: HTTPXMock):
    # GIVEN a host answering empty lists
    _mock_ok(httpx_mock, {**_SEGMENTS_ONLY, "segments": [], "speakers": []})

    # WHEN an empty word list is passed
    aai.Diarizer().diarize(b"RIFF", transcript=[])

    # THEN it is sent, not mistaken for segments-only mode
    assert _json_part(_sent(httpx_mock)["transcript"]) == {"words": []}


def test_diarize_accepts_a_sync_transcript_response(httpx_mock: HTTPXMock):
    # GIVEN a sync transcription result with timestamps
    _mock_ok(httpx_mock)
    transcript = aai.SyncTranscriptResponse(
        text="hello hi",
        words=[aai.SyncWord(**w) for w in _WORDS],
        confidence=0.92,
        audio_duration_ms=9500,
        session_id="sync-session",
    )

    # WHEN it is passed as the transcript
    aai.Diarizer().diarize(b"RIFF", transcript=transcript)

    # THEN its words are sent
    assert _json_part(_sent(httpx_mock)["transcript"]) == {"words": _WORDS}


def test_diarize_accepts_a_list_of_words(httpx_mock: HTTPXMock):
    # GIVEN a mix of SyncWord and dict words, one without timestamps
    _mock_ok(httpx_mock)
    words: List[Any] = [aai.SyncWord(text="hello", confidence=0.9), _WORDS[1]]

    # WHEN the list is passed
    aai.Diarizer().diarize(b"RIFF", transcript=words)

    # THEN the unset timings are left out for the API to fill
    assert _json_part(_sent(httpx_mock)["transcript"]) == {
        "words": [{"text": "hello", "confidence": 0.9}, _WORDS[1]]
    }


@pytest.mark.parametrize("transcript", ["hello hi", 42])
def test_diarize_rejects_other_transcript_types(transcript):
    with pytest.raises(TypeError, match="transcript must be"):
        aai.Diarizer().diarize(b"RIFF", transcript=transcript)


def test_diarize_pcm_sends_config_and_pcm_content_type(httpx_mock: HTTPXMock):
    # GIVEN raw PCM settings
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    config = aai.DiarizationConfig(sample_rate=16000, channels=1, language_code="es")

    # WHEN raw bytes are diarized
    aai.Diarizer().diarize(b"\x00\x01" * 160, config=config)

    # THEN the config precedes the audio, which is labelled audio/pcm
    parts = _sent(httpx_mock)
    assert list(parts) == ["config", "audio"]
    assert _json_part(parts["config"]) == {
        "sample_rate": 16000,
        "channels": 1,
        "language_code": "es",
    }
    assert parts["audio"].get_content_type() == "audio/pcm"
    assert parts["audio"].get_filename() == "audio.pcm"


def test_diarize_pcm_requires_rate_and_channels(tmp_path):
    path = tmp_path / "call.pcm"
    path.write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="sample_rate and channels"):
        aai.Diarizer().diarize(path)


@pytest.mark.parametrize(
    "name, content_type",
    [
        ("call.mp3", "audio/mpeg"),
        ("call.flac", "audio/flac"),
        ("call.ogg", "audio/ogg"),
        ("call.mp4", "audio/mp4"),
        ("call.webm", "audio/webm"),
        ("call.wav", "audio/wav"),
        ("call.unknown", "audio/wav"),
    ],
)
def test_diarize_infers_the_content_type_from_the_extension(
    httpx_mock: HTTPXMock, tmp_path, name, content_type
):
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    path = tmp_path / name
    path.write_bytes(b"audio")

    aai.Diarizer().diarize(str(path))

    audio = _sent(httpx_mock)["audio"]
    assert audio.get_content_type() == content_type
    assert audio.get_filename() == name


def test_diarize_names_a_file_object(httpx_mock: HTTPXMock, tmp_path):
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    path = tmp_path / "call.mp3"
    path.write_bytes(b"ID3")

    with open(path, "rb") as f:
        aai.Diarizer().diarize(f)

    assert _sent(httpx_mock)["audio"].get_content_type() == "audio/mpeg"


def test_diarize_content_type_override_wins(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)

    aai.Diarizer().diarize(io.BytesIO(b"OggS"), content_type="audio/ogg")

    assert _sent(httpx_mock)["audio"].get_content_type() == "audio/ogg"


def test_diarize_rejects_a_url():
    with pytest.raises(ValueError, match="does not accept URLs"):
        aai.Diarizer().diarize("https://example.com/call.wav")


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        aai.DiarizationConfig(speakers_expected=2)


def test_rejects_other_products_configs():
    with pytest.raises(TypeError, match="expects DiarizationConfig"):
        aai.Diarizer(config=aai.SyncTranscriptionConfig())
    with pytest.raises(TypeError, match="expects DiarizationConfig"):
        aai.Diarizer().diarize(b"RIFF", config=aai.DictationConfig())


def test_default_config_and_per_call_override(httpx_mock: HTTPXMock):
    # GIVEN a diarizer with a default config
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)
    diarizer = aai.Diarizer(config=aai.DiarizationConfig(language_code="de"))

    # WHEN one call uses it and one overrides it
    diarizer.diarize(b"RIFF")
    diarizer.diarize(b"RIFF", config=aai.DiarizationConfig(language_code="fr"))

    # THEN the override replaces the default
    first, second = httpx_mock.get_requests()
    assert _json_part(_parts(first)["config"]) == {"language_code": "de"}
    assert _json_part(_parts(second)["config"]) == {"language_code": "fr"}


def test_unauthorized_maps_to_diarization_error(httpx_mock: HTTPXMock):
    # GIVEN a missing key, which the API answers with a problem+json 401
    httpx_mock.add_response(
        url=DIARIZE_URL,
        method="POST",
        status_code=401,
        json={
            "status": 401,
            "title": "Unauthorized",
            "detail": "Missing Authorization header",
        },
        headers={"content-type": "application/problem+json"},
    )

    # WHEN diarizing, THEN a DiarizationError carries status, code and detail
    with pytest.raises(aai.DiarizationError) as exc_info:
        aai.Diarizer().diarize(b"RIFF")

    error = exc_info.value
    assert error.status_code == 401
    assert error.error_code == "unauthorized"
    assert "Missing Authorization header" in str(error)
    assert isinstance(error, aai.AssemblyAIError)
    assert not isinstance(error, aai.SyncTranscriptError)


@pytest.mark.parametrize(
    "status, title, code",
    [
        (400, "Bad Request", "bad_request"),
        (411, "Length Required", "length_required"),
        (413, "Audio Too Large", "audio_too_large"),
        (415, "Unsupported Media Type", "unsupported_media_type"),
        (504, "Inference Timeout", "inference_timeout"),
    ],
)
def test_problem_details_map_to_diarization_error(
    httpx_mock: HTTPXMock, status, title, code
):
    httpx_mock.add_response(
        url=DIARIZE_URL,
        method="POST",
        status_code=status,
        json={"status": status, "title": title, "detail": "nope"},
        headers={"content-type": "application/problem+json"},
    )

    with pytest.raises(aai.DiarizationError) as exc_info:
        aai.Diarizer().diarize(b"RIFF", transcript={"words": _WORDS})

    assert exc_info.value.status_code == status
    assert exc_info.value.error_code == code
    assert "nope" in str(exc_info.value)


def test_capacity_error_surfaces_retry_after(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=DIARIZE_URL,
        method="POST",
        status_code=503,
        json={"status": 503, "title": "Capacity Exceeded", "detail": "busy"},
        headers={"Retry-After": "2"},
    )

    with pytest.raises(aai.DiarizationError) as exc_info:
        aai.Diarizer().diarize(b"RIFF")

    assert exc_info.value.error_code == "capacity_exceeded"
    assert exc_info.value.retry_after == 2


def test_non_json_error_keeps_the_status(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=DIARIZE_URL, method="POST", status_code=502, text="Bad Gateway"
    )

    with pytest.raises(aai.DiarizationError) as exc_info:
        aai.Diarizer().diarize(b"RIFF")

    assert exc_info.value.status_code == 502
    assert "Bad Gateway" in str(exc_info.value)


def test_custom_base_url(httpx_mock: HTTPXMock):
    # GIVEN a client pointed at the EU host
    eu = "https://diarization.eu.assemblyai.com/"
    _mock_ok(
        httpx_mock,
        _SEGMENTS_ONLY,
        url="https://diarization.eu.assemblyai.com/v1/diarize",
    )
    settings = aai.Settings(api_key="eu-key", diarization_base_url=eu)

    # WHEN diarizing through it
    client = aai.Client(settings=settings)
    aai.Diarizer(client=client).diarize(b"RIFF")
    client.close()

    # THEN the request went to that host with its key
    request = httpx_mock.get_requests()[0]
    assert request.url.host == "diarization.eu.assemblyai.com"
    assert request.headers["authorization"] == "eu-key"


def test_uses_the_diarization_timeout(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)

    aai.Diarizer().diarize(b"RIFF")

    timeout = httpx_mock.get_requests()[0].extensions["timeout"]
    assert timeout["read"] == aai.settings.diarization_http_timeout


def test_api_key_constructor_builds_own_client(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock, _SEGMENTS_ONLY)

    with aai.Diarizer(api_key="per-call-key") as diarizer:
        diarizer.diarize(b"RIFF")

    assert httpx_mock.get_requests()[0].headers["authorization"] == "per-call-key"


def test_response_ignores_unknown_fields(httpx_mock: HTTPXMock):
    _mock_ok(httpx_mock, {**_SEGMENTS_ONLY, "new_field": 1})

    assert aai.Diarizer().diarize(b"RIFF").audio_duration_ms == 9500


def test_diarization_settings_defaults():
    settings = aai.Settings()

    assert settings.diarization_base_url == "https://diarization.assemblyai.com"
    assert settings.diarization_http_timeout == 330.0


def test_public_names_are_exported():
    from assemblyai import diarization
    from assemblyai.diarization import v1

    for name in v1.__all__:
        assert getattr(diarization, name) is getattr(v1, name)
    assert aai.Diarizer is v1.Diarizer
    assert aai.AsyncDiarizer is v1.AsyncDiarizer
    assert aai.DiarizationError is v1.DiarizationError
