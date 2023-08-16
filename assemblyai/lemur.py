from typing import Any, Dict, List, Optional, Union

from . import api
from . import client as _client
from . import types


class _LemurImpl:
    def __init__(
        self,
        *,
        client: _client.Client,
        sources: List[types.LemurSource],
    ) -> None:
        self._client = client

        self._sources = [types.LemurSourceRequest.from_lemur_source(s) for s in sources]

    def question(
        self,
        questions: List[types.LemurQuestion],
        context: Optional[Union[str, Dict[str, Any]]],
        timeout: Optional[float],
        final_model: Optional[types.LemurModel],
        max_output_size: Optional[int],
    ) -> types.LemurQuestionResponse:
        response = api.lemur_question(
            client=self._client.http_client,
            request=types.LemurQuestionRequest(
                sources=self._sources,
                questions=questions,
                context=context,
                final_model=final_model,
                max_output_size=max_output_size,
            ),
            http_timeout=timeout,
        )

        return response

    def summarize(
        self,
        context: Optional[Union[str, Dict[str, Any]]],
        answer_format: Optional[str],
        final_model: Optional[types.LemurModel],
        max_output_size: Optional[int],
        timeout: Optional[float],
    ) -> types.LemurSummaryResponse:
        response = api.lemur_summarize(
            client=self._client.http_client,
            request=types.LemurSummaryRequest(
                sources=self._sources,
                context=context,
                answer_format=answer_format,
                final_model=final_model,
                max_output_size=max_output_size,
            ),
            http_timeout=timeout,
        )

        return response

    def action_items(
        self,
        context: Optional[Union[str, Dict[str, Any]]],
        answer_format: Optional[str],
        final_model: Optional[types.LemurModel],
        max_output_size: Optional[int],
        timeout: Optional[float],
    ) -> types.LemurActionItemsResponse:
        response = api.lemur_action_items(
            client=self._client.http_client,
            request=types.LemurActionItemsRequest(
                sources=self._sources,
                context=context,
                answer_format=answer_format,
                final_model=final_model,
                max_output_size=max_output_size,
            ),
            http_timeout=timeout,
        )

        return response

    def task(
        self,
        prompt: str,
        final_model: Optional[types.LemurModel],
        max_output_size: Optional[int],
        timeout: Optional[float],
    ):
        response = api.lemur_task(
            client=self._client.http_client,
            request=types.LemurTaskRequest(
                sources=self._sources,
                prompt=prompt,
                final_model=final_model,
                max_output_size=max_output_size,
            ),
            http_timeout=timeout,
        )

        return response


class Lemur:
    """
    AssemblyAI's LeMUR (Leveraging Large Language Models to Understand Recognized Speech) framework
    to process audio files with an LLM.

    See https://www.assemblyai.com/docs/Models/lemur for more information.
    """

    def __init__(
        self,
        sources: List[types.LemurSource],
        client: Optional[_client.Client] = None,
    ) -> None:
        """
        Creates a new LeMUR instance to process audio files with an LLM.

        Args:

            sources: One or a list of sources to process (e.g. a `Transcript` or a `TranscriptGroup`)
            client: The client to use for the LeMUR instance. If not provided, the default client will be used
        """
        self._client = client or _client.Client.get_default()

        self._impl = _LemurImpl(
            client=self._client,
            sources=sources,
        )

    def question(
        self,
        questions: Union[types.LemurQuestion, List[types.LemurQuestion]],
        context: Optional[Union[str, Dict[str, Any]]] = None,
        final_model: Optional[types.LemurModel] = None,
        max_output_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> types.LemurQuestionResponse:
        """
        Question & Answer allows you to ask free form questions about one or many transcripts.

        This can be any question you find useful, such as judging the outcome or determining facts
        about the audio. For instance, you can ask for action items from a meeting, did the customer
        respond positively, or count how many times a word or phrase was said.

        See also Best Practices on LeMUR: https://www.assemblyai.com/docs/Guides/lemur_best_practices

        Args:
            questions: One or a list of questions to ask.
            context: The context which is shared among all questions. This can be a string or a dictionary.
            final_model: The model that is used for the final prompt after compression is performed (options: "basic" and "default").
            max_output_size: Max output size in tokens
            timeout: The timeout in seconds to wait for the answer(s).

        Returns: One or a list of answer objects.
        """

        if not isinstance(questions, list):
            questions = [questions]

        return self._impl.question(
            questions=questions,
            context=context,
            final_model=final_model,
            max_output_size=max_output_size,
            timeout=timeout,
        )

    def summarize(
        self,
        context: Optional[Union[str, Dict[str, Any]]] = None,
        answer_format: Optional[str] = None,
        final_model: Optional[types.LemurModel] = None,
        max_output_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> types.LemurSummaryResponse:
        """
        Summary allows you to distill a piece of audio into a few impactful sentences.
        You can give the model context to get more pinpoint results while outputting the
        results in a variety of formats described in human language.

        See also Best Practices on LeMUR: https://www.assemblyai.com/docs/Guides/lemur_best_practices

        Args:
            context: An optional context on the transcript.
            answer_format: The format on how the summary shall be summarized.
            final_model: The model that is used for the final prompt after compression is performed (options: "basic" and "default").
            max_output_size: Max output size in tokens
            timeout: The timeout in seconds to wait for the summary.

        Returns: The summary as a string.
        """

        return self._impl.summarize(
            context=context,
            answer_format=answer_format,
            final_model=final_model,
            max_output_size=max_output_size,
            timeout=timeout,
        )

    def action_items(
        self,
        context: Optional[Union[str, Dict[str, Any]]] = None,
        answer_format: Optional[str] = None,
        final_model: Optional[types.LemurModel] = None,
        max_output_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> types.LemurActionItemsResponse:
        """
        Action Items allows you to generate action items from one or many transcripts.

        You can provide the model with a context to get more pinpoint results while outputting the
        results in a variety of formats described in human language.

        See also Best Practices on LeMUR: https://www.assemblyai.com/docs/Guides/lemur_best_practices

        Args:
            context: An optional context on the transcript.
            answer_format: The preferred format for the result action items.
            final_model: The model that is used for the final prompt after compression is performed (options: "basic" and "default").
            max_output_size: Max output size in tokens
            timeout: The timeout in seconds to wait for the action items response.

        Returns: The action items as a string.
        """

        return self._impl.action_items(
            context=context,
            answer_format=answer_format,
            final_model=final_model,
            max_output_size=max_output_size,
            timeout=timeout,
        )

    def task(
        self,
        prompt: str,
        final_model: Optional[types.LemurModel] = None,
        max_output_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> types.LemurTaskResponse:
        """
        Task feature allows you to submit a custom prompt to the model.

        See also Best Practices on LeMUR: https://www.assemblyai.com/docs/Guides/lemur_best_practices

        Args:
            prompt: The prompt to use for this task.
            final_model: The model that is used for the final prompt after compression is performed (options: "basic" and "default").
            max_output_size: Max output size in tokens
            timeout: The timeout in seconds to wait for the task.

        Returns: A response to a question or task submitted via custom prompt (with source transcripts or other sources taken into the context)
        """

        return self._impl.task(
            prompt=prompt,
            final_model=final_model,
            max_output_size=max_output_size,
            timeout=timeout,
        )
