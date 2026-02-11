from urllib.parse import urlencode

from pytest_mock import MockFixture

from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
)


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

    params = StreamingParameters(sample_rate=16000)
    client.connect(params)

    expected_headers = {
        "sample_rate": params.sample_rate,
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

    params = StreamingParameters(sample_rate=16000)
    client.connect(params)

    expected_headers = {
        "sample_rate": params.sample_rate,
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
        end_of_turn_confidence_threshold=0.5,
        min_end_of_turn_silence_when_confident=2000,
        max_turn_silence=3000,
    )

    client.connect(params)

    expected_headers = {
        "end_of_turn_confidence_threshold": params.end_of_turn_confidence_threshold,
        "min_end_of_turn_silence_when_confident": params.min_end_of_turn_silence_when_confident,
        "max_turn_silence": params.max_turn_silence,
        "sample_rate": params.sample_rate,
    }

    assert actual_url == f"wss://api.example.com/v3/ws?{urlencode(expected_headers)}"

    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15


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

    params = StreamingParameters(sample_rate=16000)
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
        webhook_url="https://example.com/webhook",
        webhook_auth_header_name="X-Webhook-Secret",
        webhook_auth_header_value="my-secret",
    )

    client.connect(params)

    expected_params = {
        "sample_rate": params.sample_rate,
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
    assert "min_end_of_turn_silence_when_confident=200" in actual_url
    assert "prompt=Transcribe" in actual_url
    assert "keyterms_prompt=" in actual_url  # keyterms_prompt is JSON-encoded

    assert actual_additional_headers["Authorization"] == "test"
    assert actual_additional_headers["AssemblyAI-Version"] == "2025-05-12"
    assert "AssemblyAI/1.0" in actual_additional_headers["User-Agent"]

    assert actual_open_timeout == 15
