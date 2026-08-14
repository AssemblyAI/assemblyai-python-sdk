"""Tests for `api_key=` construction across the client surface.

Covers what the four transcribers, the two HTTP clients, and the two streaming
clients share: building one from an explicit key, how that interacts with an
explicit `client=`/`options=`, and the guarantee that neither the global
settings nor a caller's own options object is mutated along the way.
"""

import pytest

import assemblyai as aai
from assemblyai.streaming.v3 import (
    AsyncStreamingClient,
    StreamingClient,
    StreamingClientOptions,
)

aai.settings.api_key = "test"


@pytest.fixture
def no_global_api_key():
    """Clears the global API key, so only an explicit `api_key=` can authenticate."""

    original = aai.settings.api_key
    aai.settings.api_key = None
    yield
    aai.settings.api_key = original


# == transcribers and clients: api_key= ==


def test_transcriber_accepts_api_key(no_global_api_key):
    # When constructing a Transcriber with only an explicit key
    transcriber = aai.Transcriber(api_key="explicit-key")

    # Then the client authenticates with it and the global settings are untouched
    assert transcriber._client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_sync_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.SyncTranscriber(api_key="explicit-key")

    assert transcriber._client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


@pytest.mark.asyncio
async def test_async_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.AsyncTranscriber(api_key="explicit-key")
    try:
        # Then the client authenticates with it and is owned by the transcriber
        assert transcriber.client.settings.api_key == "explicit-key"
        assert transcriber._owns_client is True
        assert aai.settings.api_key is None
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed


@pytest.mark.asyncio
async def test_async_sync_transcriber_accepts_api_key(no_global_api_key):
    transcriber = aai.AsyncSyncTranscriber(api_key="explicit-key")
    try:
        assert transcriber.client.settings.api_key == "explicit-key"
        assert transcriber._owns_client is True
        assert aai.settings.api_key is None
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed


def _caller_settings() -> aai.Settings:
    """Settings with a distinguishable field, to prove they survive a derive."""

    return aai.Settings(api_key="from-client", http_timeout=42.5)


@pytest.mark.parametrize("transcriber_class", [aai.Transcriber, aai.SyncTranscriber])
def test_api_key_takes_precedence_over_a_given_client(transcriber_class):
    caller_client = aai.Client(settings=_caller_settings())

    transcriber = transcriber_class(client=caller_client, api_key="explicit-key")

    # Then a derived client is used, not the caller's
    assert transcriber._client is not caller_client
    assert transcriber._client.settings.api_key == "explicit-key"
    # And the caller's client is untouched, with its other settings carried over
    assert caller_client.settings.api_key == "from-client"
    assert transcriber._client.settings.http_timeout == 42.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcriber_class", [aai.AsyncTranscriber, aai.AsyncSyncTranscriber]
)
async def test_async_api_key_takes_precedence_over_a_given_client(transcriber_class):
    caller_client = aai.AsyncClient(settings=_caller_settings())

    transcriber = transcriber_class(client=caller_client, api_key="explicit-key")
    try:
        assert transcriber.client is not caller_client
        assert transcriber.client.settings.api_key == "explicit-key"
        assert caller_client.settings.api_key == "from-client"
        assert transcriber.client.settings.http_timeout == 42.5
        # And the derived client is the transcriber's to close
        assert transcriber._owns_client is True
    finally:
        await transcriber.aclose()

    assert transcriber.client.http_client.is_closed
    # The caller's client is left open for the caller to close
    assert not caller_client.http_client.is_closed

    await caller_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcriber_class", [aai.AsyncTranscriber, aai.AsyncSyncTranscriber]
)
async def test_async_client_passed_alone_stays_the_callers(transcriber_class):
    async with aai.AsyncClient(settings=aai.settings) as caller_client:
        transcriber = transcriber_class(client=caller_client)

        assert transcriber.client is caller_client
        assert transcriber._owns_client is False

        await transcriber.aclose()

        # aclose() leaves a client it does not own alone
        assert not caller_client.http_client.is_closed


def test_client_without_arguments_uses_the_global_settings():
    # When constructing a client with no arguments
    client = aai.Client()

    # Then it copies the global settings rather than sharing them
    assert client.settings.api_key == aai.settings.api_key
    assert client.settings is not aai.settings


def test_client_accepts_api_key(no_global_api_key):
    # When constructing a client with only an explicit key
    client = aai.Client(api_key="explicit-key")

    # Then it authenticates with it and the global settings are untouched
    assert client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_async_client_accepts_api_key(no_global_api_key):
    client = aai.AsyncClient(api_key="explicit-key")

    assert client.settings.api_key == "explicit-key"
    assert aai.settings.api_key is None


def test_api_key_overrides_the_key_on_given_settings():
    # Given a settings object of the caller's own
    settings = aai.Settings(api_key="from-settings")

    # When a client is built from it with an explicit key
    client = aai.Client(settings=settings, api_key="explicit-key")

    # Then the explicit key wins and the caller's settings are unchanged
    assert client.settings.api_key == "explicit-key"
    assert settings.api_key == "from-settings"


def test_missing_api_key_names_every_way_to_provide_one(no_global_api_key):
    with pytest.raises(ValueError) as exc_info:
        aai.Client()

    message = str(exc_info.value)
    assert "ASSEMBLYAI_API_KEY" in message
    assert "aai.settings.api_key" in message
    assert "api_key=" in message


# == streaming clients: api_key= ==


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_accepts_api_key(client_class):
    # When constructing with only an explicit key
    client = client_class(api_key="explicit-key")

    # Then the options carry it and every other option keeps its default
    defaults = StreamingClientOptions(api_key="explicit-key")
    assert client._options.api_key == "explicit-key"
    assert client._options.token is None
    assert client._options == defaults


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_overrides_the_key_on_options(client_class):
    options = StreamingClientOptions(api_key="from-options")

    client = client_class(options, api_key="explicit-key")

    # Then the constructor key wins
    assert client._options.api_key == "explicit-key"


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_override_carries_every_other_option(client_class):
    options = StreamingClientOptions(
        api_key="from-options",
        token="temporary-token",
        api_host="streaming.example.org",
        connect_timeout=2.5,
        max_connection_retries=7,
        connection_retry_delay=1.25,
        terminate_timeout=9.5,
    )

    client = client_class(options, api_key="explicit-key")

    # Then only api_key differs from what the caller set
    resolved = client._options
    assert resolved.api_key == "explicit-key"
    assert resolved.token == "temporary-token"
    assert resolved.api_host == "streaming.example.org"
    assert resolved.connect_timeout == 2.5
    assert resolved.max_connection_retries == 7
    assert resolved.connection_retry_delay == 1.25
    assert resolved.terminate_timeout == 9.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_api_key_override_does_not_mutate_the_callers_options(client_class):
    options = StreamingClientOptions(api_key="from-options", connect_timeout=2.5)

    client = client_class(options, api_key="explicit-key")

    # Then the caller's own object is untouched, and a copy was used
    assert options.api_key == "from-options"
    assert client._options is not options
    assert client._options.connect_timeout == 2.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_still_accepts_options_only(client_class):
    # Given the options a caller builds today, positionally
    options = StreamingClientOptions(api_key="from-options", connect_timeout=2.5)

    client = client_class(options)

    # Then the very object passed in is what the client uses
    assert client._options is options
    assert client._options.connect_timeout == 2.5


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_still_accepts_a_token_in_options(client_class):
    client = client_class(StreamingClientOptions(token="temporary-token"))

    assert client._options.token == "temporary-token"
    assert client._options.api_key is None


@pytest.mark.parametrize("client_class", [StreamingClient, AsyncStreamingClient])
def test_streaming_client_without_credentials_names_both_fixes(client_class):
    with pytest.raises(ValueError) as exc_info:
        client_class()

    message = str(exc_info.value)
    assert "api_key=" in message
    assert "StreamingClientOptions" in message
    assert "token=" in message
