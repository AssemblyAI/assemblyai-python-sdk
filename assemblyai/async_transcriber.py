from __future__ import annotations

import asyncio
import os
import stat
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urlparse

import httpx
from typing_extensions import Self

from . import async_api, types
from . import async_client as _async_client
from ._transcript_fields import TranscriptFields, config_from_response

AudioSource = Union[str, bytes, "os.PathLike[str]", BinaryIO]
"""An audio URL, a local file path, raw `bytes`, or an opened binary file."""

# Read this much per thread hop when streaming a file off disk into an upload.
_UPLOAD_CHUNK_SIZE = 1024 * 1024

# Matches the thread-pool size the sync `Transcriber` uses for group submissions.
_DEFAULT_MAX_CONCURRENCY = 8

_T = TypeVar("_T")


async def _run_in_thread(func: Callable[..., _T], *args: Any) -> _T:
    """Runs a blocking call on the default executor."""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None, func, *args)


def _open_binary(path: str, mode: str) -> BinaryIO:
    """Opens a file in binary `mode`. Call it via `_run_in_thread`."""

    return cast(BinaryIO, open(path, mode))


def _peek_length(stream: BinaryIO) -> Optional[int]:
    """
    Returns the bytes left in `stream`, or `None` if the size is unknown.

    A known size lets the upload send `Content-Length`. Pipes and sockets
    cannot report a size.
    """

    try:
        offset = stream.tell()
    except (AttributeError, OSError, ValueError):
        offset = 0

    try:
        stat_result = os.fstat(stream.fileno())
        if not stat.S_ISREG(stat_result.st_mode):
            # A pipe or terminal reports st_size 0, which is not its length.
            raise OSError
        length = stat_result.st_size
    except (AttributeError, OSError, ValueError):
        try:
            length = stream.seek(0, os.SEEK_END)
            stream.seek(offset)
        except (AttributeError, OSError, ValueError):
            return None

    return max(length - offset, 0)


async def _aiter_stream(
    stream: BinaryIO,
    chunk_size: int = _UPLOAD_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Yields `stream` in chunks. Each read runs off the event loop."""

    while True:
        chunk = await _run_in_thread(stream.read, chunk_size)
        if not chunk:
            break

        yield chunk


async def _aiter_path(
    path: str,
    chunk_size: int = _UPLOAD_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Yields the file at `path` in chunks. Opens and reads it off the loop."""

    audio_file = await _run_in_thread(_open_binary, path, "rb")
    try:
        async for chunk in _aiter_stream(audio_file, chunk_size):
            yield chunk
    finally:
        await _run_in_thread(audio_file.close)


def _upload_request(
    data: AudioSource,
) -> Tuple[Union[bytes, AsyncIterator[bytes]], Dict[str, str]]:
    """
    Turns audio into an upload body and the headers that describe it.

    Streams paths and file objects instead of reading them into memory. Sets
    `Content-Length` when the size is known, so httpx skips chunked encoding.

    Returns: `(content, headers)`.
    """

    if isinstance(data, (bytes, bytearray)):
        # httpx derives Content-Length from bytes on its own.
        return bytes(data), {}

    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        content: Union[bytes, AsyncIterator[bytes]] = _aiter_path(path)
        try:
            length: Optional[int] = os.path.getsize(path)
        except OSError:
            length = None
    elif hasattr(data, "read"):
        content = _aiter_stream(data)
        length = _peek_length(data)
    else:
        raise TypeError(f"unsupported audio input type: {type(data).__name__}")

    headers = {} if length is None else {"Content-Length": str(length)}

    return content, headers


class AsyncTranscript(TranscriptFields, types.Sourcable):
    """
    The asyncio counterpart of `Transcript`.

    Carries the same fields as `Transcript`, such as `text` and `utterances`.
    Every method that calls the API is a coroutine. An `AsyncTranscriber`
    creates these and owns the HTTP client they use.
    """

    def __init__(
        self,
        transcript_id: Optional[str],
        client: _async_client.AsyncClient,
    ) -> None:
        """
        Creates an `AsyncTranscript` for an existing transcript id.

        Args:
            transcript_id: The id of the transcript.
            client: The `AsyncClient` whose connection pool to use.
        """
        self._client = client
        self._transcript_id = transcript_id
        self._transcript: Optional[types.TranscriptResponse] = None
        self._redacted_audio_url: Optional[str] = None

    @classmethod
    def from_response(
        cls,
        *,
        client: _async_client.AsyncClient,
        response: types.TranscriptResponse,
    ) -> Self:
        self = cls(transcript_id=response.id, client=client)
        self._transcript = response

        return self

    def _response(self) -> types.TranscriptResponse:
        if not self._transcript:
            raise ValueError("The internal Transcript object is None.")

        return self._transcript

    def _require_id(self, action: str) -> str:
        if not self._transcript_id:
            raise ValueError(f"Cannot {action}. The internal transcript ID is None.")

        return self._transcript_id

    @property
    def id(self) -> Optional[str]:
        "The unique identifier of your transcription"

        return self._transcript_id

    @property
    def config(self) -> types.TranscriptionConfig:
        "Return the corresponding configurations for the given transcript."

        if self._transcript is None:
            raise ValueError(
                "Cannot access the configuration. The internal Transcript object is None."
            )

        return config_from_response(self._transcript)

    async def wait_for_completion(self) -> Self:
        """
        Polls the transcript until its status is `completed` or `error`.

        Sleeps `settings.polling_interval` seconds between polls. Other tasks
        run during the sleep.

        Returns: this `AsyncTranscript`, with the finished response.
        """

        transcript_id = self._require_id("wait for completion")

        while True:
            # No try-except - if there is an HTTP error then surface it to user
            self._transcript = await async_api.get_transcript(
                self._client.http_client,
                transcript_id,
            )

            if self._transcript.status in (
                types.TranscriptStatus.completed,
                types.TranscriptStatus.error,
            ):
                return self

            await asyncio.sleep(self._client.settings.polling_interval)

    async def export_subtitles_srt(
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

        return await async_api.export_subtitles_srt(
            client=self._client.http_client,
            transcript_id=self._require_id("export subtitles"),
            chars_per_caption=chars_per_caption,
        )

    async def export_subtitles_vtt(
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

        return await async_api.export_subtitles_vtt(
            client=self._client.http_client,
            transcript_id=self._require_id("export subtitles"),
            chars_per_caption=chars_per_caption,
        )

    async def word_search(
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

        response = await async_api.word_search(
            client=self._client.http_client,
            transcript_id=self._require_id("perform word search"),
            words=words,
        )

        return response.matches

    async def get_sentences(self) -> List[types.Sentence]:
        """
        Semantically segment your transcript into sentences to create more reader-friendly transcripts.

        Returns: A list of sentence objects.
        """

        response = await async_api.get_sentences(
            client=self._client.http_client,
            transcript_id=self._require_id("get sentences"),
        )

        return response.sentences

    async def get_paragraphs(self) -> List[types.Paragraph]:
        """
        Semantically segment your transcript into paragraphs to create more reader-friendly transcripts.

        Returns: A list of paragraph objects.
        """

        response = await async_api.get_paragraphs(
            client=self._client.http_client,
            transcript_id=self._require_id("get paragraphs"),
        )

        return response.paragraphs

    async def get_redacted_audio_url(self) -> str:
        """
        Retrieve the URL for the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`.
        Polls until the redacted audio is ready. Later calls return the cached
        URL.

        Returns: The URL of the redacted audio file.
        """

        if self._redacted_audio_url is not None:
            return self._redacted_audio_url

        if not self.config.redact_pii or not self.config.redact_pii_audio:
            raise ValueError(
                "Redacted audio is only available when `redact_pii` and `redact_pii_audio` are set to `True`."
            )

        transcript_id = self._require_id("get redacted audio url")

        while True:
            try:
                response = await async_api.get_redacted_audio(
                    client=self._client.http_client,
                    transcript_id=transcript_id,
                )
            except types.RedactedAudioIncompleteError:
                await asyncio.sleep(self._client.settings.polling_interval)
                continue

            self._redacted_audio_url = response.redacted_audio_url

            return self._redacted_audio_url

    async def save_redacted_audio(self, filepath: str) -> None:
        """
        Retrieve the PII-redacted audio file, if `redact_pii_audio` was enabled on the `TranscriptionConfig`

        Args:
            filepath: The path to save the redacted audio file to.
        """

        url = await self.get_redacted_audio_url()

        # The redacted audio lives behind a pre-signed URL, so it is fetched
        # with a bare client rather than the API-authenticated one.
        async with httpx.AsyncClient() as client:
            async with client.stream(method="GET", url=url) as response:
                if response.status_code not in (
                    httpx.codes.OK,
                    httpx.codes.NOT_MODIFIED,
                ):
                    raise types.RedactedAudioUnavailableError(
                        f"Fetching redacted audio failed with status code {response.status_code}",
                        response.status_code,
                    )

                audio_file = await _run_in_thread(_open_binary, filepath, "wb")
                try:
                    async for chunk in response.aiter_bytes():
                        await _run_in_thread(audio_file.write, chunk)
                finally:
                    await _run_in_thread(audio_file.close)


class AsyncTranscriber:
    """
    The asyncio counterpart of `Transcriber`. Transcribes URLs and local audio
    files without blocking the event loop.

    Every method that calls the API is a coroutine. Many transcriptions run
    concurrently on one thread, with no thread pool and no blocking
    `concurrent.futures.Future`.

    The transcriber owns an HTTP connection pool. Close it with `aclose()`, or
    use the transcriber as an async context manager.

    Example:
        ```python
        import asyncio
        import assemblyai as aai

        aai.settings.api_key = "your-key"

        async def main():
            async with aai.AsyncTranscriber() as transcriber:
                transcript = await transcriber.transcribe("./audio.mp3")
                print(transcript.text)

        asyncio.run(main())
        ```

        Transcribing several files concurrently is plain asyncio:
        ```python
        async with aai.AsyncTranscriber() as transcriber:
            transcripts = await asyncio.gather(
                transcriber.transcribe("./one.mp3"),
                transcriber.transcribe("./two.mp3"),
            )
        ```
    """

    def __init__(
        self,
        *,
        client: Optional[_async_client.AsyncClient] = None,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> None:
        """
        Initializes the `AsyncTranscriber` with the given parameters.

        Args:
            client: The `AsyncClient` to use. If `None`, the transcriber
                creates one from the global `settings` and closes it on
                `aclose()`. Pass a client to share one pool between
                transcribers.
            config: The default configuration for the `AsyncTranscriber`. If
                `None`, a default `TranscriptionConfig` is used.
        """
        from . import settings as default_settings

        self._owns_client = client is None
        self._client = client or _async_client.AsyncClient(settings=default_settings)
        self.config = config or types.TranscriptionConfig()

    @property
    def client(self) -> _async_client.AsyncClient:
        """The `AsyncClient` this transcriber sends requests with."""

        return self._client

    async def upload_file(self, data: AudioSource) -> str:
        """
        Uploads an audio file, given as a local path, raw `bytes`, or a binary
        object.

        Streams paths and file objects off the event loop.

        Args:
            data: A local file (as path), raw `bytes`, or a binary object.

        Returns: The URL of the uploaded audio file.
        """

        content, headers = _upload_request(data)

        return await async_api.upload_file(
            self._client.http_client,
            content,
            headers=headers,
        )

    async def submit(
        self,
        data: AudioSource,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> AsyncTranscript:
        """
        Submits a transcription job without waiting for its completion.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
            config: Transcription options and features. If `None` is given, the
                transcriber's default configuration will be used.

        Returns: The queued `AsyncTranscript`. Await `wait_for_completion()`
            to poll for the result.
        """

        config = config or self.config

        if isinstance(data, str) and urlparse(data).scheme in {"http", "https"}:
            audio_url = data
        else:
            # Note: If uploading fails, it should raise an Exception to the user.
            audio_url = await self.upload_file(data)

        request = types.TranscriptRequest(
            audio_url=audio_url,
            **config.raw.dict(exclude_none=True),
        )

        # No try-except - if there is an HTTP error raise it to the user
        response = await async_api.create_transcript(
            client=self._client.http_client,
            request=request,
        )

        return AsyncTranscript.from_response(client=self._client, response=response)

    async def transcribe(
        self,
        data: AudioSource,
        config: Optional[types.TranscriptionConfig] = None,
    ) -> AsyncTranscript:
        """
        Transcribes an audio file and waits for the result. Accepts a local
        path, a URL, raw `bytes`, or a binary object.

        Args:
            data: An URL, a local file (as path), raw `bytes`, or a binary object.
            config: Transcription options and features. If `None` is given, the
                transcriber's default configuration will be used.

        Returns: The completed `AsyncTranscript`. Check its `status`. A
            server-side failure returns `TranscriptStatus.error` and does not
            raise.
        """

        transcript = await self.submit(data=data, config=config)

        return await transcript.wait_for_completion()

    async def submit_group(
        self,
        data: List[AudioSource],
        config: Optional[types.TranscriptionConfig] = None,
        return_failures: bool = False,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> Union[
        List[AsyncTranscript],
        Tuple[List[AsyncTranscript], List[types.AssemblyAIError]],
    ]:
        """
        Submits multiple transcription jobs without waiting for their completion.

        Args:
            data: A list of local paths, URLs, raw `bytes`, or binary objects (can be mixed).
            config: Transcription options and features. If `None` is given, the
                transcriber's default configuration will be used.
            return_failures: Return the errors instead of raising the first one.
            max_concurrency: How many submissions run at once.

        Returns: The submitted transcripts, in the order of `data`. Also returns
            the errors of the failed ones when `return_failures` is set.
        """

        return await self._gather(
            data,
            lambda item: self.submit(data=item, config=config),
            return_failures=return_failures,
            max_concurrency=max_concurrency,
        )

    async def transcribe_group(
        self,
        data: List[AudioSource],
        config: Optional[types.TranscriptionConfig] = None,
        return_failures: bool = False,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> Union[
        List[AsyncTranscript],
        Tuple[List[AsyncTranscript], List[types.AssemblyAIError]],
    ]:
        """
        Transcribes a list of files and waits for all of them. Accepts local
        paths, URLs, raw `bytes`, and binary objects.

        Args:
            data: A list of local paths, URLs, raw `bytes`, or binary objects (can be mixed).
            config: Transcription options and features. If `None` is given, the
                transcriber's default configuration will be used.
            return_failures: Return the errors instead of raising the first one.
            max_concurrency: How many transcriptions run at once.

        Returns: The completed transcripts, in the order of `data`. Also returns
            the errors of the failed ones when `return_failures` is set.
        """

        return await self._gather(
            data,
            lambda item: self.transcribe(data=item, config=config),
            return_failures=return_failures,
            max_concurrency=max_concurrency,
        )

    async def _gather(
        self,
        data: List[AudioSource],
        operation: Callable[[AudioSource], Awaitable[AsyncTranscript]],
        *,
        return_failures: bool,
        max_concurrency: int,
    ) -> Union[
        List[AsyncTranscript],
        Tuple[List[AsyncTranscript], List[types.AssemblyAIError]],
    ]:
        """
        Runs `operation` over `data`, in order, with at most
        `max_concurrency` in flight.

        Awaits every item, even after an earlier failure, so no task is left
        running. A failure is never dropped: it is collected or raised once the
        batch settles.
        """

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run(item: AudioSource) -> AsyncTranscript:
            async with semaphore:
                return await operation(item)

        results = await asyncio.gather(
            *(_run(item) for item in data),
            return_exceptions=True,
        )

        transcripts: List[AsyncTranscript] = []
        failures: List[types.AssemblyAIError] = []

        for result in results:
            if isinstance(result, BaseException):
                if not return_failures or not isinstance(result, types.AssemblyAIError):
                    raise result
                failures.append(result)
            else:
                transcripts.append(result)

        if return_failures:
            return transcripts, failures

        return transcripts

    async def get_by_id(self, transcript_id: str) -> AsyncTranscript:
        """
        Fetch an existing transcript, waiting until it is completed.

        Args:
            transcript_id: the id of the transcript to fetch

        Returns: The transcript identified by the given id.
        """

        transcript = AsyncTranscript(transcript_id=transcript_id, client=self._client)

        return await transcript.wait_for_completion()

    async def delete_by_id(self, transcript_id: str) -> AsyncTranscript:
        """
        Delete an existing transcript.

        Args:
            transcript_id: the id of the transcript to delete

        Returns: The deleted transcript, with relevant fields/attributes cleared.
        """

        response = await async_api.delete_transcript(
            client=self._client.http_client,
            transcript_id=transcript_id,
        )

        return AsyncTranscript.from_response(client=self._client, response=response)

    async def list_transcripts(
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
        async with aai.AsyncTranscriber() as transcriber:
            params = aai.ListTranscriptParameters()
            page = await transcriber.list_transcripts(params)
            while page.page_details.before_id_of_prev_url is not None:
                params.before_id = page.page_details.before_id_of_prev_url
                page = await transcriber.list_transcripts(params)
        ```
        """

        return await async_api.list_transcripts(
            client=self._client.http_client,
            params=params,
        )

    async def aclose(self) -> None:
        """
        Closes the HTTP connection pool.

        Leaves a client that was passed in alone. Its creator closes it.
        """

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.aclose()
