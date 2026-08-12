"""The asyncio counterpart of `transcript.py`."""

from __future__ import annotations

import asyncio
from typing import Any, BinaryIO, Callable, List, Optional, TypeVar, cast

import httpx
from typing_extensions import Self

from ... import async_client as _async_client
from ... import types
from . import async_api
from ._base import TERMINAL_STATUSES, _BaseTranscript

_T = TypeVar("_T")


async def _run_in_thread(func: Callable[..., _T], *args: Any) -> _T:
    """Runs a blocking call on the default executor."""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(None, func, *args)


def _open_binary(path: str, mode: str) -> BinaryIO:
    """Opens a file in binary `mode`. Call it via `_run_in_thread`."""

    return cast(BinaryIO, open(path, mode))


class AsyncTranscript(_BaseTranscript):
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

    @property
    def id(self) -> Optional[str]:
        "The unique identifier of your transcription"

        return self._transcript_id

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

            if self._transcript.status in TERMINAL_STATUSES:
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
