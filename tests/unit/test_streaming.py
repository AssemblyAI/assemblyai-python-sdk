import json
import logging
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from pydantic import ValidationError
from pytest_mock import MockFixture
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.frames import Close

from assemblyai.streaming.v3 import (
    NoiseSuppressionModel,
    SpeechModel,
    SpeechStartedEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    StreamingPiiPolicy,
    StreamingPiiSubstitution,
    TurnEvent,
    Word,
)
from assemblyai.streaming.v3.models import TerminateSession


def _disable_rw_threads(mocker: MockFixture):
    """
    Disable the read and write threads for the WebSocket.
    """

    mocker.patch("threading.Thread.start", return_value=None)


def test_client_connect(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )
    client.connect(params)

    expected_headers = {
        "sample_rate": params.sample_rate,
        "speech_model": str(params.speech_model),
    }

    assert actual_url == f"wss://api.example.com/v3/ws?{urlencode(expected_headers)}"
    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15


def test_client_connect_with_token(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(token="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )
    client.connect(params)

    expected_headers = {
        "sample_rate": params.sample_rate,
        "speech_model": str(params.speech_model),
    }

    assert actual_url == f"wss://api.example.com/v3/ws?{urlencode(expected_headers)}"
    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15


def test_client_connect_all_parameters(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        end_of_turn_confidence_threshold=0.5,
        min_end_of_turn_silence_when_confident=2000,
        max_turn_silence=3000,
    )

    client.connect(params)

    expected_headers = {
        "end_of_turn_confidence_threshold": params.end_of_turn_confidence_threshold,
        "max_turn_silence": params.max_turn_silence,
        "sample_rate": params.sample_rate,
        "speech_model": str(params.speech_model),
        "min_turn_silence": params.min_end_of_turn_silence_when_confident,
    }

    assert actual_url == f"wss://api.example.com/v3/ws?{urlencode(expected_headers)}"

    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15


def test_client_connect_with_redact_pii(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        include_partial_turns=False,
        redact_pii=True,
        redact_pii_policies=[
            StreamingPiiPolicy.email_address,
            StreamingPiiPolicy.phone_number,
        ],
        redact_pii_sub=StreamingPiiSubstitution.entity_name,
    )
    client.connect(params)

    assert "include_partial_turns=False" in actual_url
    assert "redact_pii=True" in actual_url
    assert "redact_pii_sub=entity_name" in actual_url
    assert "redact_pii_policies=" in actual_url
    assert "email_address" in actual_url
    assert "phone_number" in actual_url


def test_client_connect_with_voice_focus(mocker: MockFixture):
    # Given: client + voice_focus parameters
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        voice_focus=NoiseSuppressionModel.near_field,
        voice_focus_threshold=0.5,
    )

    # When: connect
    client.connect(params)

    # Then: new wire keys are sent; old keys never appear
    assert "voice_focus=near-field" in actual_url
    assert "voice_focus_threshold=0.5" in actual_url
    assert "noise_suppression_model" not in actual_url
    assert "noise_suppression_threshold" not in actual_url


def test_noise_suppression_deprecated_alias_migrates_to_voice_focus(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: a client passing the legacy noise_suppression_* fields
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        noise_suppression_model=NoiseSuppressionModel.far_field,
        noise_suppression_threshold=0.7,
    )

    # When: connect
    with caplog.at_level(logging.WARNING):
        client.connect(params)

    # Then: legacy values migrate to voice_focus_* on the wire and a deprecation
    # warning is logged for each migrated field
    assert "voice_focus=far-field" in actual_url
    assert "voice_focus_threshold=0.7" in actual_url
    assert "noise_suppression_model" not in actual_url
    assert "noise_suppression_threshold" not in actual_url
    assert any(
        "noise_suppression_model" in r.message and "deprecated" in r.message
        for r in caplog.records
    )
    assert any(
        "noise_suppression_threshold" in r.message and "deprecated" in r.message
        for r in caplog.records
    )


def test_voice_focus_conflict_prefers_new_name(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: both legacy and new fields are set
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        voice_focus=NoiseSuppressionModel.near_field,
        voice_focus_threshold=0.4,
        noise_suppression_model=NoiseSuppressionModel.far_field,
        noise_suppression_threshold=0.9,
    )

    # When: connect
    with caplog.at_level(logging.WARNING):
        client.connect(params)

    # Then: voice_focus wins; conflict warning logged for each field
    assert "voice_focus=near-field" in actual_url
    assert "voice_focus_threshold=0.4" in actual_url
    assert "noise_suppression_model" not in actual_url
    assert "noise_suppression_threshold" not in actual_url
    assert any(
        "Both `noise_suppression_model` and `voice_focus` are set" in r.message
        for r in caplog.records
    )
    assert any(
        "Both `noise_suppression_threshold` and `voice_focus_threshold` are set"
        in r.message
        for r in caplog.records
    )


def test_api_host_accepts_ws_scheme(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="ws://127.0.0.1:8080")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )
    client.connect(params)

    assert actual_url.startswith("ws://127.0.0.1:8080/v3/ws?")
    assert "wss://ws://" not in actual_url


def test_api_host_defaults_to_wss_when_scheme_missing(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )
    client.connect(params)

    assert actual_url.startswith("wss://api.example.com/v3/ws?")


def test_deprecated_min_turn_silence_is_normalized(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        min_end_of_turn_silence_when_confident=200,
    )
    client.connect(params)

    assert "min_turn_silence=200" in actual_url
    assert "min_end_of_turn_silence_when_confident" not in actual_url


def test_min_turn_silence_conflict_prefers_new_name(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        min_end_of_turn_silence_when_confident=200,
        min_turn_silence=500,
    )
    client.connect(params)

    assert "min_turn_silence=500" in actual_url
    assert "min_turn_silence=200" not in actual_url
    assert "min_end_of_turn_silence_when_confident" not in actual_url


def test_client_send_audio(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )
    client.connect(params)
    client.stream(b"test audio data")

    assert client._write_queue.qsize() == 1
    assert isinstance(client._write_queue.get(timeout=1), bytes)


def test_client_connect_with_webhook(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        webhook_url="https://example.com/webhook",
        webhook_auth_header_name="X-Webhook-Secret",
        webhook_auth_header_value="my-secret",
    )

    client.connect(params)

    expected_params = {
        "sample_rate": params.sample_rate,
        "speech_model": str(params.speech_model),
        "webhook_url": params.webhook_url,
        "webhook_auth_header_name": params.webhook_auth_header_name,
        "webhook_auth_header_value": params.webhook_auth_header_value,
    }

    assert actual_url == f"wss://api.example.com/v3/ws?{urlencode(expected_params)}"
    assert actual_additional_headers["Authorization"] == "test"
    assert actual_open_timeout == 15


def test_client_connect_with_u3_pro_and_prompt(mocker: MockFixture):
    actual_url = None
    actual_additional_headers = None
    actual_open_timeout = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url, actual_additional_headers, actual_open_timeout
        actual_url = url
        actual_additional_headers = additional_headers
        actual_open_timeout = open_timeout

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=8000,
        speech_model=SpeechModel.u3_pro,
        min_end_of_turn_silence_when_confident=200,
        prompt="Transcribe this audio with beautiful punctuation and formatting.",
        keyterms_prompt=["yes", "no", "okay"],
    )

    client.connect(params)

    # The expected URL should contain all the parameters
    assert "sample_rate=8000" in actual_url
    assert "speech_model=u3-pro" in actual_url
    assert "min_turn_silence=200" in actual_url
    assert "min_end_of_turn_silence_when_confident" not in actual_url
    assert "prompt=Transcribe" in actual_url
    assert "keyterms_prompt=" in actual_url  # keyterms_prompt is JSON-encoded

    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15


def test_client_connect_with_speaker_labels(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        speaker_labels=True,
        max_speakers=3,
    )

    client.connect(params)

    assert "speaker_labels=True" in actual_url
    assert "max_speakers=3" in actual_url


def test_client_connect_with_continuous_partials(mocker: MockFixture):
    # Given: client + continuous_partials=True (U3-Pro steady-partials mode)
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.u3_rt_pro,
        continuous_partials=True,
    )

    # When: connect
    client.connect(params)

    # Then: parameter reaches the URL
    assert "continuous_partials=True" in actual_url


def test_customer_support_audio_capture_warns_when_enabled(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: client + customer_support_audio_capture=True
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
        customer_support_audio_capture=True,
    )

    # When: connect
    with caplog.at_level(logging.WARNING):
        client.connect(params)

    # Then: parameter reaches the URL and a warning is logged
    assert "customer_support_audio_capture=True" in actual_url
    assert any(
        "session audio" in r.message and "support" in r.message for r in caplog.records
    )


def test_customer_support_audio_capture_no_warning_when_disabled(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: client without customer_support_audio_capture
    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        pass

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )

    # When: connect
    with caplog.at_level(logging.WARNING):
        client.connect(params)

    # Then: no support-audio warning is logged
    assert not any(
        "session audio" in r.message and "support" in r.message for r in caplog.records
    )


def test_client_connect_with_whisper_rt(mocker: MockFixture):
    actual_url = None

    def mocked_websocket_connect(
        url: str, additional_headers: dict, open_timeout: float
    ):
        nonlocal actual_url
        actual_url = url

    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        new=mocked_websocket_connect,
    )

    _disable_rw_threads(mocker)

    options = StreamingClientOptions(api_key="test", api_host="api.example.com")
    client = StreamingClient(options)

    params = StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.whisper_rt,
    )

    client.connect(params)

    assert "speech_model=whisper-rt" in actual_url


def test_turn_event_with_speaker_label():
    data = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": True,
        "end_of_turn": False,
        "transcript": "Hello world",
        "end_of_turn_confidence": 0.85,
        "words": [],
        "speaker_label": "B",
    }
    event = TurnEvent.parse_obj(data)
    assert event.speaker_label == "B"


def test_turn_event_without_speaker_label():
    data = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": True,
        "end_of_turn": False,
        "transcript": "Hello world",
        "end_of_turn_confidence": 0.85,
        "words": [],
    }
    event = TurnEvent.parse_obj(data)
    assert event.speaker_label is None


def test_word_with_speaker_field():
    # Given: a Word payload that includes a per-word speaker label
    data = {
        "start": 100,
        "end": 250,
        "confidence": 0.92,
        "text": "hello",
        "word_is_final": True,
        "speaker": "A",
    }

    # When: parsed
    word = Word.parse_obj(data)

    # Then: the speaker label is preserved
    assert word.speaker == "A"


def test_word_without_speaker_field_defaults_to_none():
    # Given: a Word payload that omits the speaker label
    data = {
        "start": 100,
        "end": 250,
        "confidence": 0.92,
        "text": "hello",
        "word_is_final": True,
    }

    # When: parsed
    word = Word.parse_obj(data)

    # Then: speaker is optional → None
    assert word.speaker is None


def test_turn_event_with_word_speakers():
    # Given: a TurnEvent with two words carrying distinct per-word speaker labels
    data = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": True,
        "end_of_turn": True,
        "transcript": "Hello world",
        "end_of_turn_confidence": 0.85,
        "words": [
            {
                "start": 0,
                "end": 100,
                "confidence": 0.9,
                "text": "Hello",
                "word_is_final": True,
                "speaker": "A",
            },
            {
                "start": 110,
                "end": 200,
                "confidence": 0.9,
                "text": "world",
                "word_is_final": True,
                "speaker": "B",
            },
        ],
        "speaker_label": "A",
    }

    # When: parsed
    event = TurnEvent.parse_obj(data)

    # Then: each word's speaker is preserved
    assert [w.speaker for w in event.words] == ["A", "B"]


def test_speech_model_required():
    """Test that omitting speech_model raises a validation error."""
    with pytest.raises(ValidationError):
        StreamingParameters(sample_rate=16000)


def test_speech_started_event():
    """Test SpeechStarted event parsing (u3-rt-pro only)"""
    data = {
        "type": "SpeechStarted",
        "timestamp": 1280,
    }
    event = SpeechStartedEvent.parse_obj(data)
    assert event.type == "SpeechStarted"
    assert event.timestamp == 1280


class _FakeWebSocket:
    """Programmable sync websocket stand-in for driving StreamingClient in tests."""

    def __init__(self, recv_script, send_raises=None, send_blocks_until=None):
        self._recv_script = list(recv_script)
        self._send_raises = send_raises
        # Optional Event the test can use to hold send() until a barrier point
        # (e.g. "release send only after the read thread has reached
        # _report_server_error"), making the read+write race deterministic.
        self._send_blocks_until = send_blocks_until
        self._recv_lock = threading.Lock()
        self.close_call_count = 0
        self.send_call_count = 0
        self.sent = []

    def recv(self, timeout=None):
        with self._recv_lock:
            if not self._recv_script:
                raise TimeoutError()
            item = self._recv_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def send(self, data):
        self.send_call_count += 1
        if self._send_blocks_until is not None:
            self._send_blocks_until.wait(timeout=2.0)
        if self._send_raises is not None:
            raise self._send_raises
        self.sent.append(data)

    def close(self):
        self.close_call_count += 1


def _connect_and_wait(client, params, seed_chunks=None, timeout=2.0):
    # Prime the write queue BEFORE connect so the write thread's first get()
    # returns immediately. If we primed after connect, client.stream() can
    # short-circuit on _stop_event (set by a fast-firing close path) and the
    # write thread parks in get(timeout=1) for a full second, never reaching
    # send() — which means the read+write race the tests target never happens.
    if seed_chunks is not None:
        for chunk in seed_chunks:
            client._write_queue.put(chunk)
    client.connect(params)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        read_done = (
            not client._read_thread.is_alive()
            if client._read_thread.ident is not None
            else True
        )
        write_done = (
            not client._write_thread.is_alive()
            if client._write_thread.ident is not None
            else True
        )
        if read_done and write_done and client._stop_event.is_set():
            return
        time.sleep(0.02)


def _default_params():
    return StreamingParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )


def test_error_event_then_close_fires_only_once(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: server Error then close + a barrier that holds send() until the
    # read thread enters _report_server_error, guaranteeing a real read+write
    # race on the close (not just an artifact of recv_script ordering).
    caplog.set_level(logging.ERROR)
    error_json = json.dumps(
        {"type": "Error", "error": "Invalid API key", "error_code": 4001}
    )
    close_exc = ConnectionClosed(rcvd=Close(4001, "Not Authorized"), sent=None)
    send_gate = threading.Event()
    fake_ws = _FakeWebSocket(
        recv_script=[error_json, close_exc],
        send_raises=close_exc,
        send_blocks_until=send_gate,
    )
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )
    real_report_server_error = StreamingClient._report_server_error

    def report_server_error_then_release(self, error):
        send_gate.set()
        return real_report_server_error(self, error)

    mocker.patch.object(
        StreamingClient, "_report_server_error", report_server_error_then_release
    )
    received = []
    received_lock = threading.Lock()

    def on_error(self_, err):
        with received_lock:
            received.append(err)

    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)

    # When: connect with a primed write queue so the write thread reaches send()
    # and parks on the barrier; the read thread races when it releases the gate.
    _connect_and_wait(
        client,
        _default_params(),
        seed_chunks=[b"\x00" * 320] * 50,
    )

    # Then: exactly one on_error with the rich server-error content.
    assert len(received) == 1, (
        f"expected exactly 1 error, got {len(received)}: {received}"
    )
    assert str(received[0]) == "Invalid API key"
    assert received[0].code == 4001
    assert fake_ws.close_call_count >= 1
    assert fake_ws.send_call_count >= 1, "write thread never reached send()"
    assert not client._read_thread.is_alive()
    assert not client._write_thread.is_alive()
    error_logs = [
        rec
        for rec in caplog.records
        if "Streaming error" in rec.message and "4001" in rec.message
    ]
    close_logs = [
        rec
        for rec in caplog.records
        if "Connection closed" in rec.message and "4001" in rec.message
    ]
    assert len(error_logs) == 1, (
        f"expected exactly 1 Streaming-error log, got {len(error_logs)}"
    )
    assert error_logs[0].levelno == logging.ERROR
    assert len(close_logs) == 1, (
        f"expected exactly 1 Connection-closed log, got {len(close_logs)}"
    )
    assert close_logs[0].levelno == logging.ERROR

    client.disconnect(terminate=True)


def test_handler_exception_does_not_block_shutdown(mocker: MockFixture):
    # Given: a websocket that raises ConnectionClosed and an on_error handler that throws
    close_exc = ConnectionClosed(rcvd=Close(1011, "server error"), sent=None)
    fake_ws = _FakeWebSocket(recv_script=[close_exc], send_raises=close_exc)
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )

    def bad_handler(self_, err):
        raise RuntimeError("boom")

    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, bad_handler)

    # When: the client connects and the handler raises during error dispatch
    _connect_and_wait(client, _default_params())

    # Then: cleanup still completes — websocket closed, both worker threads exited
    assert fake_ws.close_call_count >= 1
    assert not client._read_thread.is_alive()
    assert not client._write_thread.is_alive()

    client.disconnect(terminate=True)


def test_invalid_status_401_during_connect(mocker: MockFixture):
    # Given: websocket_connect raises InvalidStatus carrying an HTTP 401 response
    response = SimpleNamespace(status_code=401)
    invalid_status = InvalidStatus(response=response)
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        side_effect=invalid_status,
    )
    start_spy = mocker.spy(threading.Thread, "start")
    received = []

    def on_error(self_, err):
        received.append(err)

    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)

    # When: connect() is called and the handshake is rejected
    client.connect(_default_params())

    # Then: a single error with code=401 is dispatched, and neither worker thread is started
    assert len(received) == 1
    assert received[0].code == 401
    assert not client._read_thread.is_alive()
    assert not client._write_thread.is_alive()
    for call in start_spy.call_args_list:
        assert call.args[0] not in (client._read_thread, client._write_thread)

    client.disconnect()


def test_clean_close_emits_no_error_or_log(
    mocker: MockFixture, caplog: pytest.LogCaptureFixture
):
    # Given: a code-1000 ConnectionClosed delivered to the read thread (exercises
    # the `streaming_error is None` short-circuit in _report_connection_closed).
    caplog.set_level(logging.DEBUG)
    clean_close = ConnectionClosed(rcvd=Close(1000, "session ended"), sent=None)
    fake_ws = _FakeWebSocket(recv_script=[clean_close], send_raises=clean_close)
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )
    received = []

    def on_error(self_, err):
        received.append(err)

    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, on_error)

    # When: the client connects and the read thread processes the clean close
    _connect_and_wait(client, _default_params())

    # Then: no on_error fires; no WARNING/ERROR-level log mentions the close.
    # The close path took the `streaming_error is None` short-circuit.
    assert received == [], f"unexpected on_error calls: {received}"
    fatal_logs = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert fatal_logs == [], (
        f"clean close should not emit WARNING/ERROR logs, got: "
        f"{[(r.levelname, r.message) for r in fatal_logs]}"
    )

    client.disconnect()


def test_report_connection_closed_suppresses_dispatch_when_server_error_flag_set(
    mocker: MockFixture,
):
    # Given: a client with _server_error_reported pre-set (simulating "read
    # thread already dispatched the rich server error").
    fake_ws = _FakeWebSocket(recv_script=[])
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    received = []
    client.on(StreamingEvents.Error, lambda c, e: received.append(e))
    client._websocket = fake_ws
    client._server_error_reported = True

    # When: _report_connection_closed runs for the trailing close
    close_exc = ConnectionClosed(
        rcvd=Close(4001, "See Error message for details"), sent=None
    )
    client._report_connection_closed(close_exc)

    # Then: no on_error dispatch (close was logged but the duplicate suppressed)
    assert received == [], (
        f"close path dispatched despite server error already reported: {received}"
    )


def test_disconnect_terminate_sends_terminate_after_stop_set(mocker: MockFixture):
    # Given: a client with a TerminateSession queued AND _stop_event already set,
    # simulating the race where disconnect(terminate=True) puts then sets stop
    # before the write loop's next get().
    fake_ws = _FakeWebSocket(recv_script=[])
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client._websocket = fake_ws
    client._write_queue.put(TerminateSession())
    client._stop_event.set()

    # When: the write loop runs (in the test thread)
    client._write_message()

    # Then: TerminateSession was sent despite stop being set; loop exited cleanly.
    assert fake_ws.send_call_count == 1, (
        f"expected exactly 1 send (the TerminateSession), got {fake_ws.send_call_count}"
    )
    assert len(fake_ws.sent) == 1
    assert '"Terminate"' in fake_ws.sent[0]


def test_write_thread_close_is_drained_by_read_thread(mocker: MockFixture):
    # Given: recv() always times out (read thread never sees its own close)
    # but send() raises ConnectionClosed, forcing the _pending_close_error
    # drain path to be the only way the user can be notified.
    close_exc = ConnectionClosed(rcvd=Close(1011, "boom"), sent=None)
    fake_ws = _FakeWebSocket(recv_script=[], send_raises=close_exc)
    mocker.patch(
        "assemblyai.streaming.v3.client.websocket_connect",
        return_value=fake_ws,
    )
    received = []
    client = StreamingClient(
        StreamingClientOptions(api_key="test", api_host="api.example.com")
    )
    client.on(StreamingEvents.Error, lambda c, e: received.append(e))

    # When: connect with seeded audio so the write thread reaches send()
    _connect_and_wait(
        client,
        _default_params(),
        seed_chunks=[b"\x00" * 320] * 5,
    )

    # Then: the read thread drained the pending close and dispatched once.
    assert fake_ws.send_call_count >= 1, "write thread never reached send()"
    assert len(received) == 1, (
        f"expected exactly 1 error from drained pending close, got: {received}"
    )
    assert received[0].code == 1011

    client.disconnect()
