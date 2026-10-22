from typing import Optional

from ... import client as _client
from . import api
from ._base import check_config, check_text, request_body, synthesize_timeout
from .api import TTSStream
from .models import TTSConfig, TTSResponse

__all__ = ["TextToSpeech"]


class TextToSpeech:
    """
    Text in, speech out: the Text-to-Speech HTTP API, one text per request.

    `synthesize()` returns the complete audio; `stream()` returns it as the
    API generates it, so playback can start long before the sentence ends.
    Voices, models and options are those of the streaming WebSocket API,
    without a session to manage: use this client for one text at a time (a
    sentence of a voice agent, a file, a preview).

    Requests go to `settings.tts_base_url`, `https://tts.us.assemblyai.com`
    by default; set `https://tts.eu.assemblyai.com` to keep them in the EU.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        tts = aai.TextToSpeech(config=aai.TTSConfig(voice_id="aai-alba"))

        with open("hello.wav", "wb") as out:
            wav = aai.TTSConfig(voice_id="aai-alba", container=aai.TTSContainer.wav)
            out.write(tts.synthesize("Hello from AssemblyAI.", wav).audio)

        with tts.stream("Streaming straight to the speakers.") as speech:
            for chunk in speech:
                play(chunk)
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[TTSConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates a `TextToSpeech`.

        Args:
            client: The HTTP client to use. Defaults to the shared default client.
            config: Default options. A per-call `config` replaces it whole,
                so repeat the voice_id in a per-call config.
            api_key: The API key to authenticate with. Builds a `Client` for
                this instance. Given alongside `client`, it takes precedence:
                a client is built from a copy of that client's settings with
                the key replaced, and the given client is left untouched.

        Raises:
            TypeError: if `config` is not a `TTSConfig`.
        """
        check_config(type(self).__name__, config)

        # Only a client built from `api_key` is owned; `client=None` is the
        # shared default client, which stays open.
        self._owns_client = api_key is not None
        self._client = _client._resolve_client(client, api_key)
        self.config = config or TTSConfig()

    @property
    def config(self) -> TTSConfig:
        """The default configuration of the `TextToSpeech`."""
        return self._config

    @config.setter
    def config(self, config: TTSConfig) -> None:
        check_config(type(self).__name__, config)

        self._config = config or TTSConfig()

    @property
    def client(self) -> _client.Client:
        """The `Client` this instance sends requests with."""

        return self._client

    def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSResponse:
        """
        Speaks `text` and returns the complete audio.

        Nothing is sent back until the audio is complete, so a long text
        takes several seconds and an API failure is always a `TTSError`,
        never a cut response. Use `stream()` when the first audio matters.

        Args:
            text: 1 to 5,000 characters.
            config: Options for this request. Replaces (does not merge with)
                the instance's default `config`.

        Returns: a `TTSResponse` with the `audio` bytes and the response headers.

        Raises:
            TTSError: on any API refusal.
        """
        check_text(text)
        check_config(type(self).__name__, config)
        body = request_body(text, config or self._config)

        return api.synthesize(
            self._client.http_client,
            base_url=self._client.settings.tts_base_url,
            body=body,
            timeout=synthesize_timeout(self._client.settings),
        )

    def stream(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSStream:
        """
        Speaks `text` and returns the audio as it is generated.

        The request is sent when the returned `TTSStream` is entered as a
        context manager or first read; entering it returns once the API has
        sent the first audio. Iterate the stream for the bytes, and use
        `with` (or call `close()`) so the connection is released.

        Args:
            text: 1 to 5,000 characters.
            config: Options for this request. Replaces (does not merge with)
                the instance's default `config`.

        Raises:
            TTSError: from the `with` or the first read, if the API refuses
                the request before any audio.
        """
        check_text(text)
        check_config(type(self).__name__, config)
        body = request_body(text, config or self._config)

        return api.stream_speech(
            self._client.http_client,
            base_url=self._client.settings.tts_base_url,
            body=body,
            timeout=self._client.settings.tts_http_timeout,
        )

    def close(self) -> None:
        """Closes the HTTP pool this instance owns. A shared `client` is left open."""

        if self._owns_client:
            self._client.http_client.close()

    def __enter__(self) -> "TextToSpeech":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
