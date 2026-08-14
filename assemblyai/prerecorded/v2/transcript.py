"""The ``Transcript`` result object for prerecorded transcription."""

from __future__ import annotations

import concurrent.futures
import functools
import time
from typing import List, Optional

import httpx
from typing_extensions import Self

from ... import client as _client
from ... import types
from . import api
from ._base import (
    TERMINAL_STATUSES,
    _BaseTranscript,
    _poll_timeout_message,
    config_from_response,
)


class _TranscriptImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
        transcript_id: Optional[str],
    ) -> None:
        self._client = client
        self.transcript_id = transcript_id

        self.transcript: Optional[types.TranscriptResponse] = None

    @property
    def config(self) -> types.TranscriptionConfig:
        "Returns the configuration from the internal Transcript object"
        if self.transcript is None:
            raise ValueError(
                "Cannot access the configuration. The internal Transcript object is None."
            )

        return config_from_response(self.transcript)

    @classmethod
    def from_response(
        cls,
        *,
        client: _client.Client,
        response: types.TranscriptResponse,
    ) -> Self:
        self = cls(
            client=client,
            transcript_id=response.id,
        )
        self.transcript = response

        return self

    def wait_for_completion(self, *, poll_timeout: Optional[float] = None) -> Self:
        """
        polls the given transcript until we have a status other than `processing` or `queued`

        Args:
            `poll_timeout`: How long to poll, in seconds. `None` polls until the
                transcript reaches a terminal status.
        """
        if not self.transcript_id:
            raise ValueError(
                "Cannot wait for completion. The internal transcript ID is None."
            )

        start = time.monotonic()

        while True:
            # No try-except - if there is an HTTP error then surface it to user
            self.transcript = api.get_transcript(
                self._client.http_client,
                self.transcript_id,
            )

            if self.transcript.status in TERMINAL_STATUSES:
                break

            if poll_timeout is not None and time.monotonic() - start >= poll_timeout:
                raise types.TranscriptError(
                    _poll_timeout_message(
                        transcript_id=self.transcript_id,
                        status=self.transcript.status,
                        poll_timeout=poll_timeout,
                    )
                )

            time.sleep(self._client.settings.polling_interval)

        return self

    def export_subtitles_srt(
        self,
        *,
        chars_per_caption: Optional[int],
    ) -> str:
        if not self.transcript or not self.transcript.id:
            raise ValueError(
                "Cannot export subtitles. The internal Transcript object is None."
            )

        return api.export_subtitles_srt(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
            chars_per_caption=chars_per_caption,
        )

    def export_subtitles_vtt(
        self,
        *,
        chars_per_caption: Optional[int],
    ) -> str:
        if not self.transcript or not self.transcript.id:
            raise ValueError(
                "Cannot export subtitles. The internal Transcript object is None."
            )

        return api.export_subtitles_vtt(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
            chars_per_caption=chars_per_caption,
        )

    def word_search(
        self,
        *,
        words: List[str],
    ) -> List[types.WordSearchMatch]:
        if not self.transcript or not self.transcript.id:
            raise ValueError(
                "Cannot perform word search. The internal Transcript object is None."
            )

        response = api.word_search(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
            words=words,
        )

        return response.matches

    def get_sentences(self) -> List[types.Sentence]:
        if not self.transcript or not self.transcript.id:
            raise ValueError(
                "Cannot get sentences. The internal Transcript object is None."
            )

        response = api.get_sentences(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
        )

        return response.sentences

    def get_paragraphs(self) -> List[types.Paragraph]:
        if not self.transcript or not self.transcript.id:
            raise ValueError(
                "Cannot get paragraphs. The internal Transcript object is None."
            )

        response = api.get_paragraphs(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
        )

        return response.paragraphs

    @functools.lru_cache
    def get_redacted_audio_url(self) -> str:
        """
        Retrieve the URL for the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`.
        Subsequent calls will return cached URL rather than requesting it from the API again.

        Returns: The URL of the redacted audio file.
        """
        if not self.config.redact_pii or not self.config.redact_pii_audio:
            raise ValueError(
                "Redacted audio is only available when `redact_pii` and `redact_pii_audio` are set to `True`."
            )

        if not self.transcript_id:
            raise ValueError(
                "Cannot get redacted audio url. The internal transcript ID is None."
            )

        while True:
            try:
                return api.get_redacted_audio(
                    client=self._client.http_client,
                    transcript_id=self.transcript_id,
                ).redacted_audio_url
            except types.RedactedAudioIncompleteError:
                time.sleep(self._client.settings.polling_interval)

    def save_redacted_audio(self, filepath: str):
        """
        Retrieve the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`

        Args:
            filepath: The path to save the redacted audio file to.
        """
        with httpx.stream(method="GET", url=self.get_redacted_audio_url()) as response:
            if response.status_code not in (httpx.codes.OK, httpx.codes.NOT_MODIFIED):
                raise types.RedactedAudioUnavailableError(
                    f"Fetching redacted audio failed with status code {response.status_code}",
                    response.status_code,
                )
            with open(filepath, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    @classmethod
    def delete_by_id(cls, transcript_id: str) -> Transcript:
        client = _client.Client.get_default()
        response = api.delete_transcript(
            client=client.http_client, transcript_id=transcript_id
        )

        return Transcript.from_response(client=client, response=response)


class Transcript(_BaseTranscript):
    """
    Transcript object to perform operations on the actual transcript.
    """

    def __init__(
        self,
        transcript_id: Optional[str],
        client: Optional[_client.Client] = None,
    ) -> None:
        self._client = client or _client.Client.get_default()

        self._impl = _TranscriptImpl(
            client=self._client,
            transcript_id=transcript_id,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor()

    def wait_for_completion(self, *, poll_timeout: Optional[float] = None) -> Self:
        """
        Polls the transcript until its status is `completed` or `error`.

        Args:
            poll_timeout: How long to poll, in seconds. `None` polls until the
                transcript reaches a terminal status.

        Raises:
            TranscriptError: if `poll_timeout` elapses first.
        """
        self._impl.wait_for_completion(poll_timeout=poll_timeout)

        return self

    def wait_for_completion_async(
        self,
        *,
        poll_timeout: Optional[float] = None,
    ) -> concurrent.futures.Future[Self]:
        return self._executor.submit(
            functools.partial(self.wait_for_completion, poll_timeout=poll_timeout)
        )

    @classmethod
    def from_response(
        cls,
        *,
        client: _client.Client,
        response: types.TranscriptResponse,
    ) -> Self:
        _impl = _TranscriptImpl.from_response(client=client, response=response)

        self = cls(
            client=client,
            transcript_id=response.id,
        )

        self._impl = _impl

        return self

    @classmethod
    def get_by_id(cls, transcript_id: str) -> Self:
        """Fetch an existing transcript. Blocks until the transcript is completed.

        Args:
            transcript_id: the id of the transcript to fetch

        Returns:
            The transcript object identified by the given id.
        """
        return cls(transcript_id=transcript_id).wait_for_completion()

    @classmethod
    def get_by_id_async(cls, transcript_id: str) -> concurrent.futures.Future[Self]:
        """Fetch an existing transcript asynchronously.

        Args:
            transcript_id: the id of the transcript to fetch

        Returns:
            A future that will resolve to the transcript object identified by the given id.
        """
        return cls(transcript_id=transcript_id).wait_for_completion_async()

    @classmethod
    def delete_by_id(cls, transcript_id: str) -> Transcript:
        """Delete an existing transcript. Blocks until the transcript is completed.

        Args:
            transcript_id: the id of the transcript to delete

        Returns:
            A transcript object identified by the given id, with relevant fields/attributes cleared.
        """
        return _TranscriptImpl.delete_by_id(transcript_id)

    @classmethod
    def delete_by_id_async(
        cls, transcript_id: str
    ) -> concurrent.futures.Future[Transcript]:
        """Delete an existing transcript asynchronously.

        Args:
            transcript_id: the id of the transcript to delete

        Returns:
            A future that will resolve to a transcript object identified by the given id, with relevant fields/attributes cleared.
        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_transcript = executor.submit(
                _TranscriptImpl.delete_by_id, transcript_id
            )
        return future_transcript

    @property
    def id(self) -> Optional[str]:
        "The unique identifier of your transcription"

        return self._impl.transcript_id

    def _response(self) -> types.TranscriptResponse:
        if not self._impl.transcript:
            raise ValueError("The internal Transcript object is None.")

        return self._impl.transcript

    def export_subtitles_srt(
        self,
        chars_per_caption: Optional[int] = None,
    ) -> str:
        """
        You can export your complete transcripts in SRT format,
        to be plugged into a video player for subtitles and closed captions.

        Args:
            chars_per_caption: To control the maximum number of characters per caption

        Returns: A string containing the all subtitles in SRT format.
        """

        return self._impl.export_subtitles_srt(
            chars_per_caption=chars_per_caption,
        )

    def export_subtitles_vtt(
        self,
        chars_per_caption: Optional[int] = None,
    ) -> str:
        """
        You can export your complete transcripts in VTT format,
        to be plugged into a video player for subtitles and closed captions.

        Args:
            chars_per_caption: To control the maximum number of characters per caption

        Returns: A string containing the all subtitles in VTT format.
        """

        return self._impl.export_subtitles_vtt(
            chars_per_caption=chars_per_caption,
        )

    def word_search(
        self,
        words: List[str],
    ) -> List[types.WordSearchMatch]:
        """
        Once a transcript has been completed, you can search through the transcript for a specific set of keywords.
        You can search for individual words, numbers, or phrases containing up to five words or numbers.

        Args:
            words: A list of words, numbers, or phrases (containing up to five words or numbers)

        Returns: A list of matches
        """

        return self._impl.word_search(
            words=words,
        )

    def get_sentences(
        self,
    ) -> List[types.Sentence]:
        """
        Semantically segment your transcript into sentences to create more reader-friendly transcripts.

        Returns: A list of sentence objects.
        """

        return self._impl.get_sentences()

    def get_paragraphs(
        self,
    ) -> List[types.Paragraph]:
        """
        Semantically segment your transcript into paragraphs to create more reader-friendly transcripts.

        Returns: A list of paragraph objects.
        """

        return self._impl.get_paragraphs()

    def get_redacted_audio_url(self) -> str:
        """
        Retrieve the URL for the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`.
        Subsequent calls will return cached URL rather than requesting it from the API again.

        Returns: The URL of the redacted audio file.
        """
        return self._impl.get_redacted_audio_url()

    def save_redacted_audio(self, filepath: str):
        """
        Retrieve the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`

        Args:
            filepath: The path to save the redacted audio file to.
        """
        return self._impl.save_redacted_audio(filepath=filepath)
