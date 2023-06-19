import datetime
import json
import uuid
from typing import Optional
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest
import websockets.exceptions
from faker import Faker
from pytest_mock import MockFixture

import assemblyai as aai

aai.settings.api_key = "test"
aai.settings.base_url = "https://api.assemblyai.com/v2"


def _disable_rw_threads(mocker: MockFixture):
    """
    Disable the read/write threads for the websocket
    """

    mocker.patch("threading.Thread.start", return_value=None)


def test_realtime_connect_has_parameters(mocker: MockFixture):
    """
    Test that the connect method has the correct parameters set
    """
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
        "assemblyai.transcriber.websocket_connect",
        new=mocked_websocket_connect,
    )
    _disable_rw_threads(mocker)

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda: None,
        on_error=lambda error: print(error),
        sample_rate=44_100,
        word_boost=["AssemblyAI"],
    )

    transcriber.connect(timeout=15.0)

    assert (
        actual_url
        == f"wss://api.assemblyai.com/v2/realtime/ws?{urlencode(dict(sample_rate=44100, word_boost=['AssemblyAI']))}"
    )
    assert actual_additional_headers == {"Authorization": aai.settings.api_key}
    assert actual_open_timeout == 15.0


def test_realtime_connect_succeeds(mocker: MockFixture):
    """
    Tests that the `RealtimeTranscriber` successfully connects to the `real-time` service.
    """
    on_error_called = False

    def on_error(error: aai.RealtimeError):
        nonlocal on_error_called
        on_error_called = True

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=on_error,
        sample_rate=44_100,
    )

    mocker.patch(
        "assemblyai.transcriber.websocket_connect",
        return_value=MagicMock(),
    )

    # mock the read/write threads
    _disable_rw_threads(mocker)

    # should pass
    transcriber.connect()

    # no errors should be called
    assert not on_error_called


def test_realtime_connect_fails(mocker: MockFixture):
    """
    Tests that the `RealtimeTranscriber` fails to connect to the `real-time` service.
    """

    on_error_called = False

    def on_error(error: aai.RealtimeError):
        nonlocal on_error_called
        on_error_called = True

        assert isinstance(error, aai.RealtimeError)
        assert "connection failed" in str(error)

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=on_error,
        sample_rate=44_100,
    )
    mocker.patch(
        "assemblyai.transcriber.websocket_connect",
        side_effect=Exception("connection failed"),
    )

    transcriber.connect()

    assert on_error_called


def test_realtime__read_succeeds(mocker: MockFixture, faker: Faker):
    """
    Tests the `_read` method of the `_RealtimeTranscriberImpl` class.
    """

    expected_transcripts = [
        aai.RealtimeFinalTranscript(
            created=faker.date_time(),
            text=faker.sentence(),
            audio_start=0,
            audio_end=1,
            confidence=1.0,
            words=[],
            punctuated=True,
            text_formatted=True,
        )
    ]

    received_transcripts = []

    def on_data(data: aai.RealtimeTranscript):
        nonlocal received_transcripts
        received_transcripts.append(data)

    transcriber = aai.RealtimeTranscriber(
        on_data=on_data,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    transcriber._impl._websocket = MagicMock()
    websocket_recv = [
        json.dumps(msg.dict(), default=str) for msg in expected_transcripts
    ]
    transcriber._impl._websocket.recv.side_effect = websocket_recv

    with pytest.raises(StopIteration):
        transcriber._impl._read()

    assert received_transcripts == expected_transcripts


def test_realtime__read_fails(mocker: MockFixture):
    """
    Tests the `_read` method of the `_RealtimeTranscriberImpl` class.
    """

    on_error_called = False

    def on_error(error: aai.RealtimeError):
        nonlocal on_error_called
        on_error_called = True

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=on_error,
        sample_rate=44_100,
    )

    transcriber._impl._websocket = MagicMock()
    error = websockets.exceptions.ConnectionClosedOK(rcvd=None, sent=None)
    transcriber._impl._websocket.recv.side_effect = error

    transcriber._impl._read()

    assert on_error_called


def test_realtime__write_succeeds(mocker: MockFixture):
    """
    Tests the `_write` method of the `_RealtimeTranscriberImpl` class.
    """
    audio_chunks = [
        bytes([1, 2, 3, 4, 5]),
        bytes([6, 7, 8, 9, 10]),
    ]

    actual_sent = []

    def mocked_send(data: str):
        nonlocal actual_sent
        actual_sent.append(data)

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    transcriber._impl._websocket = MagicMock()
    transcriber._impl._websocket.send = mocked_send
    transcriber._impl._stop_event.is_set = MagicMock(side_effect=[False, False, True])

    transcriber.stream(audio_chunks[0])
    transcriber.stream(audio_chunks[1])

    transcriber._impl._write()

    # assert that the correct data was sent (base64 encoded)
    assert len(actual_sent) == 2
    assert json.loads(actual_sent[0]) == {"audio_data": "AQIDBAU="}
    assert json.loads(actual_sent[1]) == {"audio_data": "BgcICQo="}


def test_realtime__encode_data(mocker: MockFixture):
    """
    Tests the `_encode_data` method of the `_RealtimeTranscriberImpl` class.
    """

    audio_chunks = [
        bytes([1, 2, 3, 4, 5]),
        bytes([6, 7, 8, 9, 10]),
    ]

    expected_encoded_data = [
        json.dumps({"audio_data": "AQIDBAU="}),
        json.dumps({"audio_data": "BgcICQo="}),
    ]

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    actual_encoded_data = []
    for chunk in audio_chunks:
        actual_encoded_data.append(transcriber._impl._encode_data(chunk))

    assert actual_encoded_data == expected_encoded_data


def test_realtime__handle_message_session_begins(mocker: MockFixture):
    """
    Tests the `_handle_message` method of the `_RealtimeTranscriberImpl` class
    with the `SessionBegins` message.
    """

    test_message = {
        "message_type": "SessionBegins",
        "session_id": str(uuid.uuid4()),
        "expires_at": datetime.datetime.now().isoformat(),
    }

    on_open_called = False

    def on_open(session_opened: aai.RealtimeSessionOpened):
        nonlocal on_open_called
        on_open_called = True
        assert isinstance(session_opened, aai.RealtimeSessionOpened)
        assert session_opened.session_id == uuid.UUID(test_message["session_id"])
        assert session_opened.expires_at.isoformat() == test_message["expires_at"]

    transcriber = aai.RealtimeTranscriber(
        on_open=on_open,
        on_data=lambda _: None,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    transcriber._impl._handle_message(test_message)

    assert on_open_called


def test_realtime__handle_message_partial_transcript(mocker: MockFixture):
    """
    Tests the `_handle_message` method of the `_RealtimeTranscriberImpl` class
    with the `PartialTranscript` message.
    """

    test_message = {
        "message_type": "PartialTranscript",
        "text": "hello world",
        "audio_start": 0,
        "audio_end": 1500,
        "confidence": 0.99,
        "created": datetime.datetime.now().isoformat(),
        "words": [
            {
                "text": "hello",
                "start": 0,
                "end": 500,
                "confidence": 0.99,
            },
            {
                "text": "world",
                "start": 500,
                "end": 1500,
                "confidence": 0.99,
            },
        ],
    }

    on_data_called = False

    def on_data(data: aai.RealtimePartialTranscript):
        nonlocal on_data_called
        on_data_called = True
        assert isinstance(data, aai.RealtimePartialTranscript)
        assert data.text == test_message["text"]
        assert data.audio_start == test_message["audio_start"]
        assert data.audio_end == test_message["audio_end"]
        assert data.confidence == test_message["confidence"]
        assert data.created.isoformat() == test_message["created"]
        assert data.words == [
            aai.RealtimeWord(
                text=test_message["words"][0]["text"],
                start=test_message["words"][0]["start"],
                end=test_message["words"][0]["end"],
                confidence=test_message["words"][0]["confidence"],
            ),
            aai.RealtimeWord(
                text=test_message["words"][1]["text"],
                start=test_message["words"][1]["start"],
                end=test_message["words"][1]["end"],
                confidence=test_message["words"][1]["confidence"],
            ),
        ]

    transcriber = aai.RealtimeTranscriber(
        on_data=on_data,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    transcriber._impl._handle_message(test_message)

    assert on_data_called


def test_realtime__handle_message_final_transcript(mocker: MockFixture):
    """
    Tests the `_handle_message` method of the `_RealtimeTranscriberImpl` class
    with the `FinalTranscript` message.
    """

    test_message = {
        "message_type": "FinalTranscript",
        "text": "Hello, world!",
        "audio_start": 0,
        "audio_end": 1500,
        "confidence": 0.99,
        "created": datetime.datetime.now().isoformat(),
        "punctuated": True,
        "text_formatted": True,
        "words": [
            {
                "text": "Hello,",
                "start": 0,
                "end": 500,
                "confidence": 0.99,
            },
            {
                "text": "world!",
                "start": 500,
                "end": 1500,
                "confidence": 0.99,
            },
        ],
    }

    on_data_called = False

    def on_data(data: aai.RealtimeFinalTranscript):
        nonlocal on_data_called
        on_data_called = True
        assert isinstance(data, aai.RealtimeFinalTranscript)
        assert data.text == test_message["text"]
        assert data.audio_start == test_message["audio_start"]
        assert data.audio_end == test_message["audio_end"]
        assert data.confidence == test_message["confidence"]
        assert data.created.isoformat() == test_message["created"]
        assert data.punctuated == test_message["punctuated"]
        assert data.text_formatted == test_message["text_formatted"]
        assert data.words == [
            aai.RealtimeWord(
                text=test_message["words"][0]["text"],
                start=test_message["words"][0]["start"],
                end=test_message["words"][0]["end"],
                confidence=test_message["words"][0]["confidence"],
            ),
            aai.RealtimeWord(
                text=test_message["words"][1]["text"],
                start=test_message["words"][1]["start"],
                end=test_message["words"][1]["end"],
                confidence=test_message["words"][1]["confidence"],
            ),
        ]

    transcriber = aai.RealtimeTranscriber(
        on_data=on_data,
        on_error=lambda _: None,
        sample_rate=44_100,
    )

    transcriber._impl._handle_message(test_message)

    assert on_data_called


def test_realtime__handle_message_error_message(mocker: MockFixture):
    """
    Tests the `_handle_message` method of the `_RealtimeTranscriberImpl` class
    with the error message.
    """

    test_message = {
        "error": "test error",
    }

    on_error_called = False

    def on_error(error: aai.RealtimeError):
        nonlocal on_error_called
        on_error_called = True
        assert isinstance(error, aai.RealtimeError)
        assert str(error) == test_message["error"]

    transcriber = aai.RealtimeTranscriber(
        on_data=lambda _: None,
        on_error=on_error,
        sample_rate=44_100,
    )

    transcriber._impl._handle_message(test_message)

    assert on_error_called


def test_realtime__handle_message_unknown_message(mocker: MockFixture):
    """
    Tests the `_handle_message` method of the `_RealtimeTranscriberImpl` class
    with an unknown message.
    """

    test_message = {
        "message_type": "Unknown",
    }

    on_data_called = False

    def on_data(data: aai.RealtimeTranscript):
        nonlocal on_data_called
        on_data_called = True

    on_error_called = False

    def on_error(error: aai.RealtimeError):
        nonlocal on_error_called
        on_error_called = True

    transcriber = aai.RealtimeTranscriber(
        on_data=on_data,
        on_error=on_error,
        sample_rate=44_100,
    )

    transcriber._impl._handle_message(test_message)

    assert not on_data_called
    assert not on_error_called


# TODO: create tests for the `RealtimeTranscriber.close` method
