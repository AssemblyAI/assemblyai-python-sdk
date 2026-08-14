"""The ``Transcriber`` entry point for prerecorded transcription."""

from __future__ import annotations

import concurrent.futures
import os
from typing import BinaryIO, List, Optional, Set, Tuple, Union

from ... import api as _root_api
from ... import client as _client
from ... import types
from . import api
from ._base import _BaseTranscriber, check_config, is_url
from .transcript import Transcript
from .transcript_group import TranscriptGroup

AudioSource = Union[str, bytes, bytearray, "os.PathLike[str]", BinaryIO]
"""An audio URL, a local file path, raw `bytes`, or an opened binary file."""


class _TranscriberImpl(_BaseTranscriber):
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

    def upload_file(self, data: AudioSource) -> str:
        if isinstance(data, (str, os.PathLike)):
            with open(os.fspath(data), "rb") as audio_file:
                return _root_api.upload_file(
                    client=self._client.http_client,
                    audio_file=audio_file,
                )

        if isinstance(data, (bytes, bytearray)):
            return _root_api.upload_file(
                client=self._client.http_client,
                audio_file=bytes(data),
            )

        if hasattr(data, "read"):
            return _root_api.upload_file(
                client=self._client.http_client,
                audio_file=data,
            )

        raise TypeError(f"unsupported audio input type: {type(data).__name__}")

    def transcribe_url(
        self,
        *,
        url: str,
        config: types.TranscriptionConfig,
        poll: bool,
        poll_timeout: Optional[float] = None,
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
            return transcript.wait_for_completion(poll_timeout=poll_timeout)

        return transcript

    def transcribe_file(
        self,
        *,
        data: AudioSource,
        config: types.TranscriptionConfig,
        poll: bool,
        poll_timeout: Optional[float] = None,
    ) -> Transcript:
        # Note: If uploading fails, it should raise an Exception to the user, hence no try-except here.
        audio_url = self.upload_file(data)

        return self.transcribe_url(
            url=audio_url,
            config=config,
            poll=poll,
            poll_timeout=poll_timeout,
        )

    def transcribe(
        self,
        data: AudioSource,
        config: Optional[types.TranscriptionConfig],
        poll: bool,
        poll_timeout: Optional[float] = None,
    ) -> Transcript:
        config = self._resolve_config(config)

        if isinstance(data, str) and is_url(data):
            return self.transcribe_url(
                url=data,
                config=config,
                poll=poll,
                poll_timeout=poll_timeout,
            )

        return self.transcribe_file(
            data=data,
            config=config,
            poll=poll,
            poll_timeout=poll_timeout,
        )

    def transcribe_group(
        self,
        *,
        data: List[AudioSource],
        config: Optional[types.TranscriptionConfig],
        poll: bool,
        return_failures: Optional[bool] = False,
    ) -> Union[TranscriptGroup, Tuple[TranscriptGroup, List[types.AssemblyAIError]]]:
        config = self._resolve_config(config)

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


class Transcriber(_BaseTranscriber):
    """
    A transcriber used for transcribing URLs or local audio files.
    """

    def __init__(
        self,
        *,
        client: Optional[_client.Client] = None,
        config: Optional[types.TranscriptionConfig] = None,
        max_workers: Optional[int] = None,
        api_key: Optional[str] = None,
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
            `api_key`: The API key to authenticate with. Builds a `Client` for this
                `Transcriber`. Given alongside `client`, it takes precedence: the
                `Transcriber` builds its own client from a copy of that client's
                settings with the key replaced, and the given client is left
                untouched.

        Raises:
            TypeError: if `config` is not a `TranscriptionConfig`.

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
        check_config(type(self).__name__, config)

        self._client = _client._resolve_client(client, api_key)

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

        Raises:
            TypeError: if `config` is not a `TranscriptionConfig`.
        """
        check_config(type(self).__name__, config)

        self._impl.config = config

    def upload_file(self, data: AudioSource) -> str:
        """
        Uploads an audio file which can be specified as local path or binary object.

        Args:
            `data`: A local file (as path), raw `bytes`, or a binary object.

        Returns: The URL of the uploaded audio file.
        """
        return self._impl.upload_file(data=data)

    def upload_file_async(self, data: AudioSource) -> concurrent.futures.Future[str]:
        """
        Uploads an audio file which can be specified as local path or binary object.

        Args:
            `data`: A local file (as path), raw `bytes`, or a binary object.

        Returns: The URL of the uploaded audio file.
        """
        return self._executor.submit(
            self._impl.upload_file,
            data=data,
        )

    def submit(
        self,
        data: AudioSource,
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
        data: List[AudioSource],
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
        data: AudioSource,
        config: Optional[types.TranscriptionConfig] = None,
        *,
        poll_timeout: Optional[float] = None,
    ) -> Transcript:
        """
        Transcribes an audio file which can be specified as local path, URL, raw `bytes`, or binary object.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
            poll_timeout: How long to poll for the result, in seconds. If `None` is given,
                polling continues until the transcript reaches a terminal status.

        Returns: The finished `Transcript`. A transcription the server failed to
            complete comes back with `status` `TranscriptStatus.error` and the
            reason in `error`; `text` and `words` are `None` then. Check `status`
            before reading the result.

        Raises:
            TypeError: if `config` is not a `TranscriptionConfig`.
            TranscriptError: if `poll_timeout` elapses before the transcript
                reaches a terminal status. The transcript keeps processing
                server-side; fetch it later with `Transcript.get_by_id`.
        """

        return self._impl.transcribe(
            data=data,
            config=config,
            poll=True,
            poll_timeout=poll_timeout,
        )

    def transcribe_async(
        self,
        data: AudioSource,
        config: Optional[types.TranscriptionConfig] = None,
        *,
        poll_timeout: Optional[float] = None,
    ) -> concurrent.futures.Future[Transcript]:
        """
        Transcribes an audio file which can be specified as local path, URL, raw `bytes`, or binary object.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
            config: Transcription options and features. If `None` is given, the Transcriber's
                default configuration will be used.
            poll_timeout: How long to poll for the result, in seconds. If `None` is given,
                polling continues until the transcript reaches a terminal status.

        Returns: A future resolving to the finished `Transcript`. A transcription
            the server failed to complete comes back with `status`
            `TranscriptStatus.error` and the reason in `error`; `text` and `words`
            are `None` then. Check `status` before reading the result.

        Raises:
            TypeError: if `config` is not a `TranscriptionConfig`. The future
                itself raises `TranscriptError` if `poll_timeout` elapses before
                the transcript reaches a terminal status. The transcript keeps
                processing server-side; fetch it later with
                `Transcript.get_by_id`.
        """
        check_config(type(self).__name__, config)

        return self._executor.submit(
            self._impl.transcribe,
            data=data,
            config=config,
            poll=True,
            poll_timeout=poll_timeout,
        )

    def transcribe_group(
        self,
        data: List[AudioSource],
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
        data: List[AudioSource],
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

        Raises:
            TypeError: if `config` is not a `TranscriptionConfig`.
        """
        check_config(type(self).__name__, config)

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
