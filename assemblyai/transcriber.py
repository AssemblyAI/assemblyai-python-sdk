from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse

from typing_extensions import Self

from . import api
from . import client as _client
from . import lemur, types


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

        return types.TranscriptionConfig(
            **self.transcript.dict(
                include=set(types.RawTranscriptionConfig.__fields__),
                exclude_none=True,
            )
        )

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

    def wait_for_completion(self) -> Self:
        """
        polls the given transcript until we have a status other than `processing` or `queued`
        """

        while True:
            try:
                self.transcript = api.get_transcript(
                    self._client.http_client,
                    self.transcript_id,
                )
            except Exception as exc:
                self.transcript = types.TranscriptResponse(
                    **self.transcript.dict(
                        exclude_none=True, exclude={"status", "error"}
                    ),
                    status=types.TranscriptStatus.error,
                    error=str(exc),
                )

            if self.transcript.status in (
                types.TranscriptStatus.completed,
                types.TranscriptStatus.error,
            ):
                break

            time.sleep(self._client.settings.polling_interval)

        return self

    def export_subtitles_srt(
        self,
        *,
        chars_per_caption: Optional[int],
    ) -> str:
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
        response = api.word_search(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
            words=words,
        )

        return response.matches

    def get_sentences(self) -> List[types.Sentence]:
        response = api.get_sentences(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
        )

        return response.sentences

    def get_paragraphs(self) -> List[types.Paragraph]:
        response = api.get_paragraphs(
            client=self._client.http_client,
            transcript_id=self.transcript.id,
        )

        return response.paragraphs


class Transcript:
    """
    Transcript object to perform operations on the actual transcript.
    """

    def __init__(
        self,
        *,
        transcript_id: Optional[str],
        client: Optional[_client.Client] = None,
    ) -> None:
        self._client = client or _client.Client.get_default()

        self._impl = _TranscriptImpl(
            client=self._client,
            transcript_id=transcript_id,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor()

    def wait_for_completion(self) -> Self:
        self._impl.wait_for_completion()

        return self

    def wait_for_completion_async(
        self,
    ) -> concurrent.futures.Future[Self]:
        return self._executor.submit(self.wait_for_completion)

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

    @property
    def id(self) -> Optional[str]:
        "The unique identifier of your transcription"

        return self._impl.transcript_id

    @property
    def config(self) -> types.TranscriptionConfig:
        "Return the corresponding configurations for the given transcript."

        return self._impl.config

    @property
    def audio_url(self) -> str:
        "The corresponding audio url"

        return self._impl.transcript.audio_url

    @property
    def text(self) -> Optional[str]:
        "The text transcription of your media file"

        return self._impl.transcript.text

    @property
    def summary(self) -> Optional[str]:
        "The summarization of the transcript"

        return self._impl.transcript.summary

    @property
    def chapters(self) -> Optional[List[types.Chapter]]:
        return self._impl.transcript.chapters

    @property
    def content_safety_labels(self) -> Optional[types.ContentSafetyResponse]:
        return self._impl.transcript.content_safety_labels

    @property
    def sentiment_analysis_results(self) -> Optional[List[types.Sentiment]]:
        return self._impl.transcript.sentiment_analysis_results

    @property
    def entities(self) -> Optional[List[types.Entity]]:
        return self._impl.transcript.entities

    @property
    def status(self) -> types.TranscriptStatus:
        "The current status of the transcript"

        return self._impl.transcript.status

    @property
    def error(self) -> Optional[str]:
        "The error message in case the transcription fails"

        return self._impl.transcript.error

    @property
    def words(self) -> Optional[List[types.Word]]:
        "The list of words in the transcript"

        return self._impl.transcript.words

    @property
    def utterances(self) -> Optional[List[types.Utterance]]:
        """
        When `dual_channel` or `speaker_labels` is enabled,
        a list of utterances in the transcript.
        """

        return self._impl.transcript.utterances

    @property
    def confidence(self) -> Optional[float]:
        "The confidence our model has in the transcribed text, between 0 and 1"

        return self._impl.transcript.confidence

    @property
    def audio_duration(self) -> Optional[float]:
        "The duration of the audio in seconds"

        return self._impl.transcript.audio_duration

    @property
    def webhook_status_code(self) -> Optional[int]:
        "The status code we received from your server when delivering your webhook"

        return self._impl.transcript.webhook_status_code

    @property
    def webhook_auth(self) -> Optional[bool]:
        "Whether the webhook was sent with an HTTP authentication header"

        return self._impl.transcript.webhook_auth

    @property
    def lemur(self) -> lemur.Lemur:
        """
        Access AssemblyAI's LeMUR features.
        """

        return lemur.Lemur(
            client=self._client,
            transcript_ids=[self._impl.transcript_id],
        )

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


class _TranscriptGroupImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
    ) -> None:
        self._client = client
        self.transcripts: List[Transcript] = []

    @property
    def transcript_ids(self) -> List[str]:
        return [t.id for t in self.transcripts]

    def add_transcript(self, transcript: Union[Transcript, str]) -> None:
        if isinstance(transcript, Transcript):
            self.transcripts.append(transcript)
        elif isinstance(transcript, str):
            self.transcripts.append(
                Transcript(
                    client=self._client,
                    transcript_id=transcript,
                )
            )
        else:
            raise TypeError("Unsupported type for `transcript`")

        return self

    def wait_for_completion(self) -> None:
        transcripts: List[Transcript] = []

        future_transcripts: Dict[concurrent.futures.Future[Transcript], str] = {}

        for transcript in self.transcripts:
            future = transcript.wait_for_completion_async()
            future_transcripts[future] = transcript

        finished_futures, _ = concurrent.futures.wait(future_transcripts)

        for future in finished_futures:
            transcripts.append(future.result())

        self.transcripts = transcripts


class TranscriptGroup:
    """
    A group of transcripts.

    Used when transcribing multiple transcripts at once.
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
    ) -> None:
        self._client = client or _client.Client.get_default()

        self._impl = _TranscriptGroupImpl(
            client=self._client,
        )

    @property
    def transcripts(self) -> List[Transcript]:
        """
        Returns the list of the transcripts within the `TranscriptGroup`
        """

        return self._impl.transcripts

    def __iter__(self) -> Iterator[Transcript]:
        """
        Iterate over the transcripts within the `TranscriptGroup`
        """

        return iter(self.transcripts)

    @property
    def status(self) -> types.TranscriptStatus:
        """
        Return the status of the `TranscriptGroup`.

        e.g. if any of the transcripts is in `error` status, the whole `TranscriptGroup` will be in `error` status.
        """

        all_status = {t.status for t in self.transcripts}

        if any(s == types.TranscriptStatus.queued for s in all_status):
            return types.TranscriptStatus.queued
        elif any(s == types.TranscriptStatus.processing for s in all_status):
            return types.TranscriptStatus.processing
        elif any(s == types.TranscriptStatus.error for s in all_status):
            return types.TranscriptStatus.error
        elif all(s == types.TranscriptStatus.completed for s in all_status):
            return types.TranscriptStatus.completed

    @property
    def lemur(self) -> lemur.Lemur:
        """
        Access AssemblyAI's LeMUR functionality.
        """

        return lemur.Lemur(
            client=self._impl._client,
            transcript_ids=self._impl.transcript_ids,
        )

    def add_transcript(
        self,
        transcript: Union[Transcript, str],
    ) -> Self:
        """
        Adds a transcript to the given `TranscriptGroup`

        Args:
            transcript: A `Transcript` object or the ID as a `str`
        """
        self._impl.add_transcript(transcript)

        return self

    def wait_for_completion(self) -> Self:
        """
        Polls each transcript within the `TranscriptGroup`.

        """
        self._impl.wait_for_completion()

        return self


class _TranscriberImpl:
    """
    Implementation of the Transcriber class.
    """

    def __init__(
        self,
        *,
        client: _client.Client,
        config: types.TranscriptionConfig,
    ) -> None:
        self._client = client
        self.config = config

    def transcribe_url(
        self,
        *,
        url: str,
        config: types.TranscriptionConfig,
        poll: bool,
    ) -> Transcript:
        transcript_request = types.TranscriptRequest(
            audio_url=url,
            **config.raw.dict(exclude_none=True),
        )
        try:
            transcript = Transcript.from_response(
                client=self._client,
                response=api.create_transcript(
                    client=self._client.http_client,
                    request=transcript_request,
                ),
            )
        except Exception as exc:
            return Transcript.from_response(
                client=self._client,
                response=types.TranscriptResponse(
                    audio_url=url,
                    **config.raw.dict(exclude_none=True),
                    status=types.TranscriptStatus.error,
                    error=str(exc),
                ),
            )

        if poll:
            return transcript.wait_for_completion()

        return transcript

    def transcribe_file(
        self,
        *,
        path: str,
        config: types.TranscriptionConfig,
        poll: bool,
    ) -> Transcript:
        with open(path, "rb") as audio_file:
            try:
                audio_url = api.upload_file(
                    client=self._client.http_client,
                    audio_file=audio_file,
                )
            except Exception as exc:
                return Transcript.from_response(
                    client=self._client,
                    response=types.TranscriptResponse(
                        audio_url=path,
                        **config.raw.dict(exclude_none=True),
                        status=types.TranscriptStatus.error,
                        error=str(exc),
                    ),
                )

        return self.transcribe_url(
            url=audio_url,
            config=config,
            poll=poll,
        )

    def transcribe(
        self,
        data: str,
        config: Optional[types.TranscriptionConfig],
        poll: bool,
    ) -> Transcript:
        if config is None:
            config = self.config

        if urlparse(data).scheme in {"http", "https"}:
            return self.transcribe_url(
                url=data,
                config=config,
                poll=poll,
            )

        return self.transcribe_file(
            path=data,
            config=config,
            poll=poll,
        )

    def transcribe_group(
        self,
        *,
        data: List[str],
        config: Optional[types.TranscriptionConfig],
        poll: bool,
    ) -> TranscriptGroup:
        if config is None:
            config = self.config

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        future_transcripts: Dict[concurrent.futures.Future[Transcript], str] = {}

        for d in data:
            transcript_future = executor.submit(
                self.transcribe,
                data=d,
                config=config,
                poll=False,
            )

            future_transcripts[transcript_future] = d

        finished_futures, _ = concurrent.futures.wait(future_transcripts)

        transcript_group = TranscriptGroup(
            client=self._client,
        )

        for future in finished_futures:
            transcript_group.add_transcript(future.result())

        if poll:
            return transcript_group.wait_for_completion()

        return transcript_group


class Transcriber:
    """
    A transcriber used for transcribing URLs or local audio files.
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[types.TranscriptionConfig] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initializes the `Transcriber` with the given parameters.

        Args:
            `client`: The `Client` to use for the `Transcriber`. If `None` is given, the
                default settings for the `Client` will be used.
            `config`: The default configuration for the `Transcriber`. If `None` is given,
                the default configuration of a `TranscriptionConfig` will be used.
            `max_workers`: The maximum number of parallel jobs when using the `_async`
                methods on the `Transcriber`. By default it uses `os.cpu_count() - 1`

        Example:
            To use the `Transcriber` with the default settings, you can simply do:
            ```
            transcriber = aai.Transcriber()
            ```

            To use the `Transcriber` with a custom configuration, you can do:
            ```
            config = aai.TranscriptionConfig(punctuate=False, format_text=False)

            transcriber = aai.Transcriber(config=config)
            ```
        """
        self._client = client or _client.Client.get_default()

        self._impl = _TranscriberImpl(
            client=self._client,
            config=config or types.TranscriptionConfig(),
        )

        if not max_workers:
            max_workers = max(1, os.cpu_count() - 1)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )

    @property
    def config(self) -> types.TranscriptionConfig:
        """
        Returns the default configuration of the `Transcriber`.
        """
        return self._impl.config

    @config.setter
    def config(self, config: types.TranscriptionConfig) -> None:
        """
        Sets the default configuration of the `Transcriber`.

        Args:
            `config`: The new default configuration.
        """
        self._impl.config = config

    def submit(
        self,
        data: str,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> Transcript:
        """
        Submits a transcription job without waiting for its completion.

        Args:
            data: An URL or a local file (as path)
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """
        return self._impl.transcribe(
            data=data,
            config=config,
            poll=False,
        )

    def transcribe(
        self,
        data: str,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> Transcript:
        """
        Transcribes an audio file whose location can be specified via a URL or file path.

        Args:
            data: An URL or a local file (as path)
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """

        return self._impl.transcribe(
            data=data,
            config=config,
            poll=True,
        )

    def transcribe_async(
        self,
        data: str,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> concurrent.futures.Future[Transcript]:
        """
        Transcribes an audio file whose location can be specified via a URL or file path.

        Args:
            data: An URL or a local file (as path)
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """

        return self._executor.submit(
            self._impl.transcribe,
            data=data,
            config=config,
            poll=True,
        )

    def transcribe_group(
        self,
        data: List[str],
        config: Optional[types.TranscriptionConfig] = None,
    ) -> TranscriptGroup:
        """
        Transcribes a list of files (as paths) or URLs with the given configs.

        Args:
            data: A list of paths or URLs (can be mixed)
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """

        return self._impl.transcribe_group(
            data=data,
            config=config,
            poll=True,
        )

    def transcribe_group_async(
        self,
        data: List[str],
        config: Optional[types.TranscriptionConfig] = None,
    ) -> concurrent.futures.Future[TranscriptGroup]:
        """
        Transcribes a list of files (as paths) or URLs with the given configs asynchronously
        by returning a `concurrent.futures.Future[TranscriptGroup]` object.

        Args:
            data: A list of paths or URLs (can be mixed)
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """

        return self._executor.submit(
            self._impl.transcribe_group,
            data=data,
            config=config,
            poll=True,
        )
