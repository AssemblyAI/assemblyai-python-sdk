from typing import Any, Dict, List, Optional, Union

from . import api
from . import client as _client
from . import types


class _LemurImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
        transcript_ids: List[str],
    ) -> None:
        self._client = client
        self._transcript_ids = transcript_ids

    def question(
        self,
        questions: List[types.LemurQuestion],
        model: types.LemurModel,
        timeout: Optional[float],
    ) -> List[types.LemurQuestionResult]:
        response = api.lemur_question(
            client=self._client.http_client,
            request=types.LemurQuestionRequest(
                transcript_ids=self._transcript_ids,
                questions=questions,
                model=model,
            ),
            http_timeout=timeout,
        )

        return response.response

    def summarize(
        self,
        context: Optional[Union[str, Dict[str, Any]]],
        answer_format: Optional[str],
        timeout: Optional[float],
    ) -> str:
        response = api.lemur_summarize(
            client=self._client.http_client,
            request=types.LemurSummaryRequest(
                transcript_ids=self._transcript_ids,
                context=context,
                answer_format=answer_format,
            ),
            http_timeout=timeout,
        )

        return response.response

    def ask_coach(
        self,
        context: Union[str, Dict[str, Any]],
        timeout: Optional[float],
    ) -> str:
        response = api.lemur_coach(
            client=self._client.http_client,
            request=types.LemurCoachRequest(
                transcript_ids=self._transcript_ids,
                context=context,
            ),
            http_timeout=timeout,
        )

        return response.response


class Lemur:
    def __init__(
        self,
        transcript_ids: List[str],
        client: Optional[_client.Client] = None,
    ) -> None:
        self._client = client or _client.Client.get_default()

        self._impl = _LemurImpl(
            client=self._client,
            transcript_ids=transcript_ids,
        )

    def question(
        self,
        questions: Union[types.LemurQuestion, List[types.LemurQuestion]],
        model: types.LemurModel = types.LemurModel.default,
        timeout: Optional[float] = None,
    ) -> Union[types.LemurQuestionResult, List[types.LemurQuestionResult]]:
        """
        Ask questions about transcripts powered by AssemblyAI's LeMUR model.

        Args:
            questions: One or a list of questions to ask.
            model: The LeMUR model to use for the question(s).
            timeout: The timeout in seconds to wait for the answer(s).

        Returns: One or a list of answer objects.
        """

        if not isinstance(questions, list):
            questions = [questions]

        answer = self._impl.question(
            questions=questions,
            model=model,
            timeout=timeout,
        )

        return answer[0] if len(answer) == 1 else answer

    def summarize(
        self,
        context: Optional[Union[str, Dict[str, Any]]] = None,
        answer_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Summarize the given transcript with the power of AssemblyAI's Lemur model.

        Args:
            context: An optional context on the transcript.
            answer_format: The format on how the summary shall be summarized.
            timeout: The timeout in seconds to wait for the summary.

        Returns: The summary as a string.
        """

        return self._impl.summarize(
            context=context,
            answer_format=answer_format,
            timeout=timeout,
        )

    def ask_coach(
        self,
        context: Union[str, Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> str:
        """
        Ask the AI coach a question on the transcript powered by AssemblyAI's Lemur model.

        Args:
            context: The context and/or question to ask the coach.
            timeout: The timeout in seconds to wait for the coach.

        Returns: The feedback of the AI's coach as a string.
        """

        return self._impl.ask_coach(
            context=context,
            timeout=timeout,
        )
