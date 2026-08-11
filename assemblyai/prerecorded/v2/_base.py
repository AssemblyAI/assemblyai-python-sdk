"""Sync/async-agnostic core for the prerecorded v2 product.

Houses the pieces that are exactly the same for the threaded ``Transcriber``
and the asyncio ``AsyncTranscriber``:

- ``TranscriptFields``, the read-only view over a fetched ``TranscriptResponse``.
- ``_BaseTranscript`` and ``_BaseTranscriber``, the bases both concurrency
  models inherit.
- ``_raise_for_status``, so both transports raise the same error per endpoint.

Subclasses implement the I/O. Sync subclasses use plain methods; async
subclasses use ``async def``. That return-type divergence is why the I/O
methods are not ``@abstractmethod`` here.
"""

from typing import Dict, List, Optional, Type, Union
from urllib.parse import urlparse

import httpx

from ... import api as _root_api
from ... import types

# The statuses a transcript stops changing at. Both pollers loop until one.
TERMINAL_STATUSES = (
    types.TranscriptStatus.completed,
    types.TranscriptStatus.error,
)


def _raise_for_status(
    response: httpx.Response,
    message: str,
    error_type: Type[types.AssemblyAIError] = types.TranscriptError,
) -> None:
    """
    Raises `error_type` unless the response is a 200.

    Args:
        `response`: the HTTP response
        `message`: what failed, e.g. `failed to retrieve transcript abc`. The
            server error is appended to it.
        `error_type`: the exception class to raise.
    """

    if response.status_code != httpx.codes.OK:
        raise error_type(
            f"{message}: {_root_api._get_error_message(response)}",
            response.status_code,
        )


def is_url(data: str) -> bool:
    """Reports whether `data` is an HTTP audio URL rather than a local path."""

    return urlparse(data).scheme in {"http", "https"}


def config_from_response(
    response: types.TranscriptResponse,
) -> types.TranscriptionConfig:
    """Rebuilds the `TranscriptionConfig` a transcript was created with."""

    return types.TranscriptionConfig(
        **response.dict(
            include=set(types.RawTranscriptionConfig.__fields__),
            exclude_none=True,
        )
    )


class TranscriptFields:
    """
    Exposes the fields of a fetched transcript.

    Subclasses implement `_response()`. Every accessor here reads it and never
    performs I/O.
    """

    def _response(self) -> types.TranscriptResponse:
        """
        Returns the fetched transcript response.

        Raises:
            ValueError: if the transcript has not been fetched yet.
        """

        raise NotImplementedError

    @property
    def json_response(self) -> Optional[dict]:
        "The full JSON response associated with the transcript."

        return self._response().dict()

    @property
    def audio_url(self) -> str:
        "The corresponding audio url"

        return self._response().audio_url

    @property
    def speech_model(self) -> Optional[str]:
        "The speech model used for the transcription"

        return self._response().speech_model

    @property
    def speech_model_used(self) -> Optional[str]:
        "The actual speech model that was used for the transcription"

        return self._response().speech_model_used

    @property
    def text(self) -> Optional[str]:
        "The text transcription of your media file"

        return self._response().text

    @property
    def translated_texts(self) -> Optional[Dict[str, str]]:
        "The translated texts transcription of your media file"

        return self._response().translated_texts

    @property
    def speech_understanding(self) -> Optional[types.SpeechUnderstandingResponse]:
        "The speech understanding results for your media file"

        return self._response().speech_understanding

    @property
    def summary(self) -> Optional[str]:
        "The summarization of the transcript"

        return self._response().summary

    @property
    def chapters(self) -> Optional[List[types.Chapter]]:
        "The list of auto-chapters results"

        return self._response().chapters

    @property
    def content_safety(self) -> Optional[types.ContentSafetyResponse]:
        "The results from the content safety analysis"

        return self._response().content_safety_labels

    @property
    def sentiment_analysis(self) -> Optional[List[types.Sentiment]]:
        "The list of sentiment analysis results"

        return self._response().sentiment_analysis_results

    @property
    def entities(self) -> Optional[List[types.Entity]]:
        "The list of entity detection results"

        return self._response().entities

    @property
    def iab_categories(self) -> Optional[types.IABResponse]:
        "The results from the IAB category detection"

        return self._response().iab_categories_result

    @property
    def auto_highlights(self) -> Optional[types.AutohighlightResponse]:
        "The results from the auto-highlights model"

        return self._response().auto_highlights_result

    @property
    def status(self) -> types.TranscriptStatus:
        "The current status of the transcript"

        return self._response().status

    @property
    def error(self) -> Optional[str]:
        "The error message in case the transcription fails"

        return self._response().error

    @property
    def words(self) -> Optional[List[types.Word]]:
        "The list of words in the transcript"

        return self._response().words

    @property
    def utterances(self) -> Optional[List[types.Utterance]]:
        """
        When `dual_channel` or `speaker_labels` is enabled,
        a list of utterances in the transcript.
        """

        return self._response().utterances

    @property
    def unredacted_text(self) -> Optional[str]:
        "The unredacted transcript text, when `redact_pii_return_unredacted` was enabled."

        return self._response().unredacted_text

    @property
    def unredacted_words(self) -> Optional[List[types.Word]]:
        "The unredacted list of words, when `redact_pii_return_unredacted` was enabled."

        return self._response().unredacted_words

    @property
    def unredacted_utterances(self) -> Optional[List[types.Utterance]]:
        "The unredacted list of utterances, when `redact_pii_return_unredacted` was enabled."

        return self._response().unredacted_utterances

    @property
    def confidence(self) -> Optional[float]:
        "The confidence our model has in the transcribed text, between 0 and 1"

        return self._response().confidence

    @property
    def audio_duration(self) -> Optional[int]:
        "The duration of the audio in seconds"

        return self._response().audio_duration

    @property
    def webhook_status_code(self) -> Optional[int]:
        "The status code we received from your server when delivering your webhook"

        return self._response().webhook_status_code

    @property
    def webhook_auth(self) -> Optional[bool]:
        "Whether the webhook was sent with an HTTP authentication header"

        return self._response().webhook_auth

    @property
    def language_code(self) -> Optional[Union[str, types.LanguageCode]]:
        "The language code of the transcript"

        return self._response().language_code

    @property
    def language_codes(self) -> Optional[List[Union[str, types.LanguageCode]]]:
        "The list of language codes for multilingual/code-switching audio"

        return self._response().language_codes


class _BaseTranscript(TranscriptFields):
    """
    Shared base for `Transcript` and `AsyncTranscript`.

    Subclasses provide the fetched response through `_response()` and the id
    through `id`. Everything here is free of I/O, so both concurrency models
    reuse it unchanged.
    """

    @property
    def id(self) -> Optional[str]:
        "The unique identifier of your transcription"

        raise NotImplementedError

    @property
    def config(self) -> types.TranscriptionConfig:
        "Return the corresponding configurations for the given transcript."

        try:
            response = self._response()
        except ValueError as error:
            # Keeps the message the flat `transcriber.py` module raised here.
            raise ValueError(f"Cannot access the configuration. {error}") from None

        return config_from_response(response)

    def _require_id(self, action: str) -> str:
        """
        Returns the transcript id, or raises when the transcript has none.

        Args:
            `action`: what the caller wanted to do, e.g. `get sentences`.
        """

        transcript_id = self.id
        if not transcript_id:
            raise ValueError(f"Cannot {action}. The internal transcript ID is None.")

        return transcript_id


class _BaseTranscriber:
    """
    Shared base for `Transcriber` and `AsyncTranscriber`.

    Subclasses provide `config` and implement the I/O.
    """

    config: types.TranscriptionConfig

    def _resolve_config(
        self,
        config: Optional[types.TranscriptionConfig],
    ) -> types.TranscriptionConfig:
        """Returns the per-call config, or the transcriber's default."""

        return config if config is not None else self.config
