"""The ``TranscriptGroup`` collection over several prerecorded transcripts."""

from __future__ import annotations

import concurrent.futures
from typing import Iterator, List, Optional, Set, Tuple, Union

from typing_extensions import Self

from ... import client as _client
from ... import types
from .transcript import Transcript


class _TranscriptGroupImpl:
    def __init__(
        self,
        *,
        transcript_ids: List[str],
        client: _client.Client,
    ) -> None:
        self._client = client
        self.transcripts: List[Transcript] = []

        for transcript_id in transcript_ids:
            self.add_transcript(transcript_id)

    @property
    def transcript_ids(self) -> List[str]:
        if any(t.id is None for t in self.transcripts):
            raise ValueError("All transcripts must have a transcript ID.")
        return [
            t.id for t in self.transcripts if t.id
        ]  # include the if check for mypy type checker

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

    def wait_for_completion(
        self, return_failures
    ) -> Union[None, List[types.AssemblyAIError]]:
        transcripts: List[Transcript] = []
        failures: List[types.AssemblyAIError] = []

        future_transcripts: Set[concurrent.futures.Future[Transcript]] = set()

        for transcript in self.transcripts:
            future = transcript.wait_for_completion_async()
            future_transcripts.add(future)

        finished_futures, _ = concurrent.futures.wait(future_transcripts)

        for future in finished_futures:
            try:
                transcripts.append(future.result())
            except types.TranscriptError as e:
                failures.append(e)

        self.transcripts = transcripts

        if return_failures is True:
            return failures
        return None


class TranscriptGroup:
    """
    A group of transcripts.

    Used when transcribing multiple transcripts at once.
    """

    def __init__(
        self,
        transcript_ids: List[str] = [],
        client: Optional[_client.Client] = None,
    ) -> None:
        self._client = client or _client.Client.get_default()

        self._impl = _TranscriptGroupImpl(
            transcript_ids=transcript_ids,
            client=self._client,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor()

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

    @classmethod
    def get_by_ids(
        cls, transcript_ids: List[str]
    ) -> Union[Self, Tuple[Self, List[types.AssemblyAIError]]]:
        return cls(transcript_ids=transcript_ids).wait_for_completion()

    @classmethod
    def get_by_ids_async(
        cls, transcript_ids: List[str]
    ) -> concurrent.futures.Future[
        Union[Self, Tuple[Self, List[types.AssemblyAIError]]]
    ]:
        return cls(transcript_ids=transcript_ids).wait_for_completion_async()

    @property
    def status(self) -> types.TranscriptStatus:
        """
        Return the status of the `TranscriptGroup`.

        e.g. if any of the transcripts is in `error` status, the whole `TranscriptGroup` will be in `error` status.
        """

        all_status = {t.status for t in self.transcripts}

        if any(s == types.TranscriptStatus.error for s in all_status):
            return types.TranscriptStatus.error
        elif any(s == types.TranscriptStatus.queued for s in all_status):
            return types.TranscriptStatus.queued
        elif any(s == types.TranscriptStatus.processing for s in all_status):
            return types.TranscriptStatus.processing
        elif all(s == types.TranscriptStatus.completed for s in all_status):
            return types.TranscriptStatus.completed
        else:
            raise ValueError(f"Unexpected status type: {all_status}")

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

    def wait_for_completion(
        self,
        return_failures: Optional[bool] = False,
    ) -> Union[Self, Tuple[Self, List[types.AssemblyAIError]]]:
        """
        Polls each transcript within the `TranscriptGroup`.

        Note - if an HTTP error is encountered when waiting for a Transcript in the TranscriptGroup, it will be popped from the group and added to the list of failures.
        You can return this list of failures with `return_failures=True`.

        Args:
            return_failures: Whether to return a list of errors for transcripts that failed due to HTTP errors.
        """
        if return_failures is True:
            failures = self._impl.wait_for_completion(return_failures=return_failures)
            if failures is None:
                raise ValueError("return_failures was set but failures object is None")
            return self, failures

        self._impl.wait_for_completion(return_failures=return_failures)

        return self

    def wait_for_completion_async(
        self,
        return_failures: Optional[bool] = False,
    ) -> concurrent.futures.Future[
        Union[Self, Tuple[Self, List[types.AssemblyAIError]]],
    ]:
        return self._executor.submit(
            self.wait_for_completion, return_failures=return_failures
        )
