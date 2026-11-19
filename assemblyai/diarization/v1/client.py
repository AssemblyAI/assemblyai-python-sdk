from typing import Optional

from ... import client as _client
from . import api
from ._base import AudioInput, TranscriptInput, check_config, resolve_request
from .models import DiarizationConfig, DiarizationResponse

__all__ = ["Diarizer"]


class Diarizer:
    """
    Speaker labels for audio: the Diarization API, one file per request.

    Pass the audio with a transcript, such as a `SyncTranscriber` result
    requested with `timestamps=True`, to get each word's speaker and the
    utterances. Leave the transcript out to get only the speaker turns
    (`segments`) and `speakers`, so diarization can run alongside
    transcription instead of after it.

    Requests go to `settings.diarization_base_url`,
    `https://diarization.assemblyai.com` by default; set
    `https://diarization.us.assemblyai.com` or
    `https://diarization.eu.assemblyai.com` to keep them in the US or EU.

    Example:
        ```python
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        transcript = aai.SyncTranscriber().transcribe(
            "call.wav", config=aai.SyncTranscriptionConfig(timestamps=True)
        )
        result = aai.Diarizer().diarize("call.wav", transcript=transcript)
        for utterance in result.utterances:
            print(f"{utterance.speaker}: {utterance.text}")
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[DiarizationConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Creates a `Diarizer`.

        Args:
            client: The HTTP client to use. Defaults to the shared default
                client. A client you pass stays yours to close.
            config: Default options. A per-call `config` replaces it whole.
            api_key: The API key to authenticate with. Builds a `Client` for
                this instance. Given alongside `client`, it takes precedence:
                a client is built from a copy of that client's settings with
                the key replaced, and the given client is left untouched.

        Raises:
            TypeError: if `config` is not a `DiarizationConfig`.
        """
        check_config(type(self).__name__, config)

        # Only a client built from `api_key` is owned; `client=None` is the
        # shared default client, which stays open.
        self._owns_client = api_key is not None
        self._client = _client._resolve_client(client, api_key)
        self.config = config or DiarizationConfig()

    @property
    def config(self) -> DiarizationConfig:
        """The default configuration of the `Diarizer`."""
        return self._config

    @config.setter
    def config(self, config: DiarizationConfig) -> None:
        check_config(type(self).__name__, config)

        self._config = config or DiarizationConfig()

    @property
    def client(self) -> _client.Client:
        """The `Client` this instance sends requests with."""

        return self._client

    def diarize(
        self,
        audio: AudioInput,
        transcript: Optional[TranscriptInput] = None,
        config: Optional[DiarizationConfig] = None,
        *,
        content_type: Optional[str] = None,
    ) -> DiarizationResponse:
        """
        Labels the speakers in `audio`.

        Args:
            audio: a local file path, raw bytes, or a binary file object. Not
                a URL.
            transcript: the words to label: a `SyncTranscriptResponse`
                (requested with `timestamps=True`), a dict with a `words`
                list, or a list of words, each with `text`, `start` and
                `end` in milliseconds. `None` runs segments-only mode:
                `words` and `utterances` come back empty.
            config: Options for this request. Replaces (does not merge with)
                the instance's default `config`.
            content_type: the audio's Content-Type, which selects the
                decoder. Inferred from the file extension when absent:
                `audio/wav`, `audio/pcm` (`.pcm`/`.raw`, or whenever
                `sample_rate`/`channels` are set), `audio/mpeg`,
                `audio/flac`, `audio/ogg`, `audio/mp4`, `audio/webm`; WAV
                when there is no known extension.

        Returns: a `DiarizationResponse`.

        Raises:
            DiarizationError: on any API refusal.
            ValueError: for a URL, or raw PCM missing `sample_rate` or `channels`.
            TypeError: for an unsupported audio or transcript type.
        """
        data, filename, audio_content_type, transcript_json, config_json = (
            resolve_request(
                type(self).__name__,
                audio,
                transcript,
                config or self._config,
                content_type,
            )
        )

        return api.diarize(
            self._client.http_client,
            base_url=self._client.settings.diarization_base_url,
            audio=data,
            filename=filename,
            audio_content_type=audio_content_type,
            transcript=transcript_json,
            config=config_json,
            timeout=self._client.settings.diarization_http_timeout,
        )

    def close(self) -> None:
        """
        Closes the HTTP pool this instance owns. A `client` you passed is
        left open.
        """

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "Diarizer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
