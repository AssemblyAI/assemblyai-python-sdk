"""The ``Transcriber`` entry point for prerecorded transcription."""

from __future__ import annotations

import concurrent.futures
import os
from typing import BinaryIO, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from ... import api as _root_api
from ... import client as _client
from ... import types
from . import api
from .transcript import Transcript
from .transcript_group import TranscriptGroup


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

    def upload_file(self, data: Union[str, bytes, BinaryIO]) -> str:
        if isinstance(data, str):
            with open(data, "rb") as audio_file:
                return _root_api.upload_file(
                    client=self._client.http_client,
                    audio_file=audio_file,
                )
        else:
            return _root_api.upload_file(
                client=self._client.http_client,
                audio_file=data,
            )

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
        # No try-except - if there is an HTTP error raise it to the user
        transcript = Transcript.from_response(
            client=self._client,
            response=api.create_transcript(
                client=self._client.http_client,
                request=transcript_request,
            ),
        )

        if poll:
            return transcript.wait_for_completion()

        return transcript

    def transcribe_file(
        self,
        *,
        data: Union[str, bytes, BinaryIO],
        config: types.TranscriptionConfig,
        poll: bool,
    ) -> Transcript:
        # Note: If uploading fails, it should raise an Exception to the user, hence no try-except here.
        audio_url = self.upload_file(data)

        return self.transcribe_url(
            url=audio_url,
            config=config,
            poll=poll,
        )

    def transcribe(
        self,
        data: Union[str, bytes, BinaryIO],
        config: Optional[types.TranscriptionConfig],
        poll: bool,
    ) -> Transcript:
        if config is None:
            config = self.config

        if isinstance(data, str) and urlparse(data).scheme in {"http", "https"}:
            return self.transcribe_url(
                url=data,
                config=config,
                poll=poll,
            )

        return self.transcribe_file(
            data=data,
            config=config,
            poll=poll,
        )

    def transcribe_group(
        self,
        *,
        data: List[Union[str, bytes, BinaryIO]],
        config: Optional[types.TranscriptionConfig],
        poll: bool,
        return_failures: Optional[bool] = False,
    ) -> Union[TranscriptGroup, Tuple[TranscriptGroup, List[types.AssemblyAIError]]]:
        if config is None:
            config = self.config

        future_transcripts: Set[concurrent.futures.Future[Transcript]] = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for d in data:
                transcript_future = executor.submit(
                    self.transcribe,
                    data=d,
                    config=config,
                    poll=False,
                )

                future_transcripts.add(transcript_future)

        finished_futures, _ = concurrent.futures.wait(future_transcripts)

        transcript_group = TranscriptGroup(
            client=self._client,
        )
        failures: List[types.AssemblyAIError] = []

        for future in finished_futures:
            try:
                transcript_group.add_transcript(future.result())
            except types.TranscriptError as e:
                failures.append(e)

        if poll is True and return_failures is True:
            res = transcript_group.wait_for_completion(return_failures=return_failures)
            if not isinstance(res, tuple):
                raise ValueError(
                    "return_failures was set but did not receive failures object"
                )
            transcript_group, completion_failures = res
            failures.extend(completion_failures)
        elif poll:
            res = transcript_group.wait_for_completion(return_failures=return_failures)
            if not isinstance(res, TranscriptGroup):
                raise ValueError(
                    "return_failures was not set but did receive failures object"
                )
            transcript_group = res

        if return_failures is True:
            return transcript_group, failures
        else:
            return transcript_group

    def list_transcripts(
        self,
        params: Optional[types.ListTranscriptParameters],
    ) -> types.ListTranscriptResponse:
        return api.list_transcripts(client=self._client.http_client, params=params)


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
            cpu_count = os.cpu_count()
            if not cpu_count:
                max_workers = 1
            else:
                max_workers = max(1, cpu_count - 1)

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

    def upload_file(self, data: Union[str, bytes, BinaryIO]) -> str:
        """
        Uploads an audio file which can be specified as local path or binary object.

        Args:
            `data`: A local file (as path), or a binary object.

        Returns: The URL of the uploaded audio file.
        """
        return self._impl.upload_file(data=data)

    def upload_file_async(
        self, data: Union[str, bytes, BinaryIO]
    ) -> concurrent.futures.Future[str]:
        """
        Uploads an audio file which can be specified as local path or binary object.

        Args:
            `data`: A local file (as path), or a binary object.

        Returns: The URL of the uploaded audio file.
        """
        return self._executor.submit(
            self._impl.upload_file,
            data=data,
        )

    def submit(
        self,
        data: Union[str, bytes, BinaryIO],
        config: Optional[types.TranscriptionConfig] = None,
    ) -> Transcript:
        """
        Submits a transcription job without waiting for its completion.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
        """
        return self._impl.transcribe(
            data=data,
            config=config,
            poll=False,
        )

    def submit_group(
        self,
        data: List[Union[str, bytes, BinaryIO]],
        config: Optional[types.TranscriptionConfig] = None,
        return_failures: Optional[bool] = False,
    ) -> Union[TranscriptGroup, Tuple[TranscriptGroup, List[types.AssemblyAIError]]]:
        """
        Submits multiple transcription jobs without waiting for their completion.

        Args:
            data: A list of local paths, URLs, raw `bytes`, or binary objects (can be mixed).
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
            return_failures: Whether to include a list of errors for transcriptions that failed due to HTTP errors
        """
        return self._impl.transcribe_group(
            data=data,
            config=config,
            poll=False,
            return_failures=return_failures,
        )

    def transcribe(
        self,
        data: Union[str, bytes, BinaryIO],
        config: Optional[types.TranscriptionConfig] = None,
    ) -> Transcript:
        """
        Transcribes an audio file which can be specified as local path, URL, raw `bytes`, or binary object.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
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
        data: Union[str, bytes, BinaryIO],
        config: Optional[types.TranscriptionConfig] = None,
    ) -> concurrent.futures.Future[Transcript]:
        """
        Transcribes an audio file which can be specified as local path, URL, raw `bytes`, or binary object.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
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
        data: List[Union[str, bytes, BinaryIO]],
        config: Optional[types.TranscriptionConfig] = None,
        return_failures: Optional[bool] = False,
    ) -> Union[TranscriptGroup, Tuple[TranscriptGroup, List[types.AssemblyAIError]]]:
        """
        Transcribes a list of files (as local paths, URLs, or binary objects).

        Args:
            data: A list of local paths, URLs, raw `bytes`, or binary objects (can be mixed).
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
            return_failures: Whether to include a list of errors for transcriptions that failed due to HTTP errors
        """

        return self._impl.transcribe_group(
            data=data,
            config=config,
            poll=True,
            return_failures=return_failures,
        )

    def transcribe_group_async(
        self,
        data: List[Union[str, bytes, BinaryIO]],
        config: Optional[types.TranscriptionConfig] = None,
        return_failures: Optional[bool] = False,
    ) -> concurrent.futures.Future[
        Union[TranscriptGroup, Tuple[TranscriptGroup, List[types.AssemblyAIError]]]
    ]:
        """
        Transcribes a list of files (as local paths, URLs, or binary objects) asynchronously.

        Args:
            data: A list of local paths, URLs, raw `bytes`, or binary objects (can be mixed).
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
            return_failures: Whether to include a list of errors for transcriptions that failed due to HTTP errors
        """

        return self._executor.submit(
            self._impl.transcribe_group,
            data=data,
            config=config,
            poll=True,
            return_failures=return_failures,
        )

    def list_transcripts(
        self,
        params: Optional[types.ListTranscriptParameters] = None,
    ) -> types.ListTranscriptResponse:
        """
        Retrieve a list of transcripts that were created. Transcripts are sorted from newest to oldest.

        Args:
            params: The parameters to filter the transcript list by.

        Returns: A page with a list of transcripts along with page details.

        To paginate over all pages, you can set the `ListTranscriptParameters.before_id`
        to the `before_id` of the `prev_url`. Example:
        ```
        transcriber = aai.Transcriber()
        params = aai.ListTranscriptParameters()
        page = transcriber.list_transcripts(params)
        while page.page_details.before_id_of_prev_url is not None:
            params.before_id = page.page_details.before_id_of_prev_url
            page = transcriber.list_transcripts(params)
        ```
        """
        return self._impl.list_transcripts(params=params)

    def list_transcripts_async(
        self,
        params: Optional[types.ListTranscriptParameters] = None,
    ) -> concurrent.futures.Future[types.ListTranscriptResponse]:
        """
        Retrieve a list of transcripts that were created. Transcripts are sorted from newest to oldest.

        Args:
            params: The parameters to filter the transcript list by.

        Returns: A page with a list of transcripts along with page details.
        """
        return self._executor.submit(self._impl.list_transcripts, params=params)
