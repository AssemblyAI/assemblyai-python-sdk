import uuid

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import (
    ENDPOINT_LEMUR,
    ENDPOINT_LEMUR_BASE,
)
from tests.unit import factories

aai.settings.api_key = "test"


def test_lemur_single_question_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether asking a single question succeeds.
    """

    # create a mock response of a LemurQuestionResponse
    mock_lemur_answer = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()

    # we only want to mock one answer
    mock_lemur_answer["response"] = [mock_lemur_answer["response"][0]]

    # prepare the question to be asked
    question = aai.LemurQuestion(
        question="Which cars do the callers want to buy?",
        context="Callers are interested in buying cars",
        answer_options=["Toyota", "Honda", "Ford", "Chevrolet"],
    )

    # update the mock question with the question
    mock_lemur_answer["response"][0]["question"] = question.question

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    transcript = aai.Transcript(str(uuid.uuid4()))

    # mimic the usage of the SDK
    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result = lemur.question(question)

    # check whether answer is not a list
    assert isinstance(result, aai.LemurQuestionResponse)

    answers = result.response

    # check the response
    assert answers[0].question == mock_lemur_answer["response"][0]["question"]
    assert answers[0].answer == mock_lemur_answer["response"][0]["answer"]

    assert result.usage.input_tokens == mock_lemur_answer["usage"]["input_tokens"]
    assert result.usage.output_tokens == mock_lemur_answer["usage"]["output_tokens"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_single_question_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether asking a single question succeeds with input text.
    """

    # create a mock response of a LemurQuestionResponse
    mock_lemur_answer = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()

    # we only want to mock one answer
    mock_lemur_answer["response"] = [mock_lemur_answer["response"][0]]

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    # prepare the question to be asked
    question = aai.LemurQuestion(
        question="Which cars do the callers want to buy?",
        context="Callers are interested in buying cars",
        answer_options=["Toyota", "Honda", "Ford", "Chevrolet"],
    )
    # test input_text input
    # mimic the usage of the SDK
    lemur = aai.Lemur()
    result = lemur.question(
        question, input_text="This transcript is a test transcript."
    )

    # check whether answer is not a list
    assert isinstance(result, aai.LemurQuestionResponse)

    answers = result.response

    # check the response
    assert answers[0].question == mock_lemur_answer["response"][0]["question"]
    assert answers[0].answer == mock_lemur_answer["response"][0]["answer"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_multiple_question_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether asking multiple questions succeeds.
    """

    # create a mock response of a LemurQuestionResponse
    mock_lemur_answer = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()

    # prepare the questions to be asked
    questions = [
        aai.LemurQuestion(
            question="Which cars do the callers want to buy?",
        ),
        aai.LemurQuestion(
            question="What price range are the callers looking for?",
        ),
    ]

    # update the mock questions with the questions
    mock_lemur_answer["response"][0]["question"] = questions[0].question
    mock_lemur_answer["response"][1]["question"] = questions[1].question

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    transcript = aai.Transcript(str(uuid.uuid4()))

    # mimic the usage of the SDK
    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result = lemur.question(questions=questions)

    assert isinstance(result, aai.LemurQuestionResponse)

    answers = result.response
    # check whether answers is a list
    assert isinstance(answers, list)

    # check the response
    for idx, answer in enumerate(answers):
        assert answer.question == mock_lemur_answer["response"][idx]["question"]
        assert answer.answer == mock_lemur_answer["response"][idx]["answer"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_multiple_question_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether asking multiple questions succeeds.
    """

    # create a mock response of a LemurQuestionResponse
    mock_lemur_answer = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()

    # prepare the questions to be asked
    questions = [
        aai.LemurQuestion(
            question="Which cars do the callers want to buy?",
        ),
        aai.LemurQuestion(
            question="What price range are the callers looking for?",
        ),
    ]

    # update the mock questions with the questions
    mock_lemur_answer["response"][0]["question"] = questions[0].question
    mock_lemur_answer["response"][1]["question"] = questions[1].question

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    # test input_text input
    # mimic the usage of the SDK
    lemur = aai.Lemur()
    result = lemur.question(
        questions, input_text="This transcript is a test transcript."
    )
    assert isinstance(result, aai.LemurQuestionResponse)

    answers = result.response
    # check whether answers is a list
    assert isinstance(answers, list)

    # check the response
    for idx, answer in enumerate(answers):
        assert answer.question == mock_lemur_answer["response"][idx]["question"]
        assert answer.answer == mock_lemur_answer["response"][idx]["answer"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_question_fails(httpx_mock: HTTPXMock):
    """
    Tests whether asking a question fails.
    """

    # prepare the question to be asked
    question = aai.LemurQuestion(
        question="Which cars do the callers want to buy?",
        context="Callers are interested in buying cars",
        answer_options=["Toyota", "Honda", "Ford", "Chevrolet"],
    )

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])

    with pytest.raises(aai.LemurError):
        lemur.question(question)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR succeeds.
    """

    # create a mock response of a LemurSummaryResponse
    mock_lemur_summary = factories.generate_dict_factory(
        factories.LemurSummaryResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/summary",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_summary,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result = lemur.summarize(context="Callers asking for cars", answer_format="TLDR")

    assert isinstance(result, aai.LemurSummaryResponse)

    summary = result.response

    # check the response
    assert summary == mock_lemur_summary["response"]

    assert result.usage.input_tokens == mock_lemur_summary["usage"]["input_tokens"]
    assert result.usage.output_tokens == mock_lemur_summary["usage"]["output_tokens"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR succeeds with input text.
    """

    # create a mock response of a LemurSummaryResponse
    mock_lemur_summary = factories.generate_dict_factory(
        factories.LemurSummaryResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/summary",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_summary,
    )

    # test input_text input
    lemur = aai.Lemur()
    result = lemur.summarize(
        context="Callers asking for cars", answer_format="TLDR", input_text="Test test"
    )

    assert isinstance(result, aai.LemurSummaryResponse)

    summary = result.response

    # check the response
    assert summary == mock_lemur_summary["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_fails(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/summary",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])

    with pytest.raises(aai.LemurError):
        lemur.summarize(context="Callers asking for cars", answer_format="TLDR")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_action_items_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether generating action items for a transcript via LeMUR succeeds.
    """

    # create a mock response of a LemurActionItemsResponse
    mock_lemur_action_items = factories.generate_dict_factory(
        factories.LemurActionItemsResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/action-items",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_action_items,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result = lemur.action_items(
        context="Customers asking for help with resolving their problem",
        answer_format="Three bullet points",
    )

    assert isinstance(result, aai.LemurActionItemsResponse)

    action_items = result.response

    # check the response
    assert action_items == mock_lemur_action_items["response"]

    assert result.usage.input_tokens == mock_lemur_action_items["usage"]["input_tokens"]
    assert (
        result.usage.output_tokens == mock_lemur_action_items["usage"]["output_tokens"]
    )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_action_items_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether generating action items for a transcript via LeMUR succeeds.
    """

    # create a mock response of a LemurActionItemsResponse
    mock_lemur_action_items = factories.generate_dict_factory(
        factories.LemurActionItemsResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/action-items",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_action_items,
    )

    # test input_text input
    lemur = aai.Lemur()
    result = lemur.action_items(
        context="Customers asking for help with resolving their problem",
        answer_format="Three bullet points",
        input_text="Test test",
    )

    assert isinstance(result, aai.LemurActionItemsResponse)

    action_items = result.response

    # check the response
    assert action_items == mock_lemur_action_items["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_action_items_fails(httpx_mock: HTTPXMock):
    """
    Tests whether generating action items for a transcript via LeMUR fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/action-items",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])

    with pytest.raises(aai.LemurError):
        lemur.action_items(
            context="Customers asking for help with resolving their problem",
            answer_format="Three bullet points",
        )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_task_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds.
    """

    # create a mock response of a LemurTaskResponse
    mock_lemur_task_response = factories.generate_dict_factory(
        factories.LemurTaskResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_task_response,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(
        sources=[aai.LemurSource(transcript)],
    )
    result = lemur.task(
        prompt="Create action items of the meeting", context="An important meeting"
    )

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert result.response == mock_lemur_task_response["response"]

    assert (
        result.usage.input_tokens == mock_lemur_task_response["usage"]["input_tokens"]
    )
    assert (
        result.usage.output_tokens == mock_lemur_task_response["usage"]["output_tokens"]
    )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_task_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds.
    """

    # create a mock response of a LemurTaskResponse
    mock_lemur_task_response = factories.generate_dict_factory(
        factories.LemurTaskResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_task_response,
    )
    # test input_text input
    lemur = aai.Lemur()
    result = lemur.task(
        prompt="Create action items of the meeting", input_text="Test test"
    )

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert result.response == mock_lemur_task_response["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize(
    "final_model",
    (
        aai.LemurModel.claude3_5_sonnet,
        aai.LemurModel.claude3_opus,
        aai.LemurModel.claude3_haiku,
        aai.LemurModel.claude3_sonnet,
        aai.LemurModel.claude2_1,
        aai.LemurModel.claude2_0,
        aai.LemurModel.default,
        aai.LemurModel.mistral7b,
    ),
)
def test_lemur_task_succeeds(final_model, httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds with other models.
    """

    # create a mock response of a LemurTaskResponse
    mock_lemur_task_response = factories.generate_dict_factory(
        factories.LemurTaskResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_task_response,
    )
    # test input_text input
    lemur = aai.Lemur()
    result = lemur.task(
        final_model=final_model,
        prompt="Create action items of the meeting",
        context="An important meeting",
        input_text="Test test",
    )

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert result.response == mock_lemur_task_response["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_ask_coach_fails(httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])

    with pytest.raises(aai.LemurError):
        lemur.task(prompt="Create action items of the meeting")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_purge_request_data_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether LeMUR request purging succeeds.
    """

    # create a mock response of a LemurPurgeResponse
    mock_lemur_purge_response = factories.generate_dict_factory(
        factories.LemurPurgeResponse
    )()

    mock_request_id: str = str(uuid.uuid4())

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{mock_request_id}",
        status_code=httpx.codes.OK,
        method="DELETE",
        json=mock_lemur_purge_response,
    )

    # mimic the usage of the SDK
    result = aai.Lemur.purge_request_data(request_id=mock_request_id)

    # check the response
    assert isinstance(result, aai.LemurPurgeResponse)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_purge_request_data_fails(httpx_mock: HTTPXMock):
    """
    Tests whether LeMUR request purging fails.
    """

    # create a mock response of a LemurPurgeResponse
    mock_lemur_purge_response = factories.generate_dict_factory(
        factories.LemurPurgeResponse
    )()

    mock_request_id: str = str(uuid.uuid4())

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{mock_request_id}",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="DELETE",
        json=mock_lemur_purge_response,
    )

    with pytest.raises(aai.LemurError):
        aai.Lemur.purge_request_data(mock_request_id)

    assert len(httpx_mock.get_requests()) == 1


def test_lemur_single_question_async_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether asking a single question succeeds when async is used.
    """

    # create a mock response of a LemurQuestionResponse
    mock_lemur_answer = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()

    # we only want to mock one answer
    mock_lemur_answer["response"] = [mock_lemur_answer["response"][0]]

    # prepare the question to be asked
    question = aai.LemurQuestion(
        question="Which cars do the callers want to buy?",
        context="Callers are interested in buying cars",
        answer_options=["Toyota", "Honda", "Ford", "Chevrolet"],
    )

    # update the mock question with the question
    mock_lemur_answer["response"][0]["question"] = question.question

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    transcript = aai.Transcript(str(uuid.uuid4()))

    # mimic the usage of the SDK
    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result_future = lemur.question_async(question)

    result = result_future.result()

    # check whether answer is not a list
    assert isinstance(result, aai.LemurQuestionResponse)

    answers = result.response

    # check the response
    assert answers[0].question == mock_lemur_answer["response"][0]["question"]
    assert answers[0].answer == mock_lemur_answer["response"][0]["answer"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_async_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR succeeds
    when async is used.
    """

    # create a mock response of a LemurSummaryResponse
    mock_lemur_summary = factories.generate_dict_factory(
        factories.LemurSummaryResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/summary",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_summary,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result_future = lemur.summarize_async(
        context="Callers asking for cars", answer_format="TLDR"
    )

    result = result_future.result()

    assert isinstance(result, aai.LemurSummaryResponse)

    summary = result.response

    # check the response
    assert summary == mock_lemur_summary["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_action_items_async_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether generating action items for a transcript via LeMUR succeeds
    when async is used.
    """

    # create a mock response of a LemurActionItemsResponse
    mock_lemur_action_items = factories.generate_dict_factory(
        factories.LemurActionItemsResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/action-items",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_action_items,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(sources=[aai.LemurSource(transcript)])
    result_future = lemur.action_items_async(
        context="Customers asking for help with resolving their problem",
        answer_format="Three bullet points",
    )

    result = result_future.result()

    assert isinstance(result, aai.LemurActionItemsResponse)

    action_items = result.response

    # check the response
    assert action_items == mock_lemur_action_items["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_task_async_succeeds_transcript(httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds when async is used.
    """

    # create a mock response of a LemurTaskResponse
    mock_lemur_task_response = factories.generate_dict_factory(
        factories.LemurTaskResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_task_response,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(
        sources=[aai.LemurSource(transcript)],
    )
    result_future = lemur.task_async(prompt="Create action items of the meeting")

    result = result_future.result()

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert result.response == mock_lemur_task_response["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_purge_request_data_async_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether LeMUR request purging succeeds when async is used...
    """

    # create a mock response of a LemurPurgeResponse
    mock_lemur_purge_response = factories.generate_dict_factory(
        factories.LemurPurgeResponse
    )()

    mock_request_id: str = str(uuid.uuid4())

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{mock_request_id}",
        status_code=httpx.codes.OK,
        method="DELETE",
        json=mock_lemur_purge_response,
    )

    # mimic the usage of the SDK
    result_future = aai.Lemur.purge_request_data_async(request_id=mock_request_id)

    result = result_future.result()

    # check the response
    assert isinstance(result, aai.LemurPurgeResponse)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_usage_data(httpx_mock: HTTPXMock):
    """
    Tests whether usage data is correctly returned.
    """

    # create a mock response of a LemurTaskResponse
    mock_lemur_task_response = factories.generate_dict_factory(
        factories.LemurTaskResponse
    )()
    mock_lemur_task_response["usage"]["input_tokens"] = 100
    mock_lemur_task_response["usage"]["output_tokens"] = 200

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR}/task",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_task_response,
    )

    # mimic the usage of the SDK
    transcript = aai.Transcript(str(uuid.uuid4()))

    lemur = aai.Lemur(
        sources=[aai.LemurSource(transcript)],
    )
    result = lemur.task(prompt="Create action items of the meeting")

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert (
        result.usage.input_tokens == mock_lemur_task_response["usage"]["input_tokens"]
    )
    assert (
        result.usage.output_tokens == mock_lemur_task_response["usage"]["output_tokens"]
    )

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_get_response_data_string_response(httpx_mock: HTTPXMock):
    """
    Tests whether a LeMUR string response data is correctly returned.
    """
    request_id = "1234"

    mock_lemur_response = factories.generate_dict_factory(
        factories.LemurStringResponse
    )()
    mock_lemur_response["request_id"] = request_id

    # mock the specific endpoint
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{request_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_lemur_response,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur()
    result = lemur.get_response_data(request_id)

    # check the response
    assert isinstance(result, aai.LemurStringResponse)
    assert result.request_id == request_id

    # test the request field is populated correctly
    assert result.request is not None
    assert hasattr(result.request, "request_endpoint")
    assert hasattr(result.request, "temperature")
    assert hasattr(result.request, "final_model")
    assert hasattr(result.request, "max_output_size")
    assert hasattr(result.request, "created_at")

    # test usage field
    assert result.usage is not None
    assert hasattr(result.usage, "input_tokens")
    assert hasattr(result.usage, "output_tokens")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_get_response_data_question_answer(httpx_mock: HTTPXMock):
    """
    Tests whether a LeMUR question-answer response data is correctly returned with questions field.
    """
    request_id = "qa-1234"

    mock_lemur_response = factories.generate_dict_factory(
        factories.LemurQuestionResponse
    )()
    mock_lemur_response["request"] = factories.generate_dict_factory(
        factories.LemurQuestionRequestDetails
    )()
    mock_lemur_response["request_id"] = request_id

    # mock the specific endpoint
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{request_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_lemur_response,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur()
    result = lemur.get_response_data(request_id)

    # check the response
    assert isinstance(result, aai.LemurQuestionResponse)
    assert result.request_id == request_id

    # test the request field is populated correctly
    assert result.request is not None
    assert result.request.request_endpoint == "/lemur/v3/question-answer"
    assert hasattr(result.request, "temperature")
    assert hasattr(result.request, "final_model")
    assert hasattr(result.request, "max_output_size")
    assert hasattr(result.request, "created_at")

    # test question-answer specific request fields
    assert hasattr(result.request, "questions")
    assert isinstance(result.request.questions, list)
    assert len(result.request.questions) == 2

    # test that questions have the right structure - one with answer_format, one with answer_options
    question1, question2 = result.request.questions
    assert hasattr(question1, "answer_format")
    assert hasattr(question1, "context")
    assert question1.question == "What is the main topic?"
    assert question1.answer_format == "short sentence"
    assert question1.context == "Meeting context"

    assert hasattr(question2, "answer_options")
    assert question2.question == "What is the sentiment?"
    assert question2.answer_options == ["positive", "negative", "neutral"]

    # test that qa-specific fields are None for other operation types
    assert result.request.prompt is None  # task-specific field
    assert result.request.context is None  # summary/action_items-specific field
    assert result.request.answer_format is None  # summary/action_items-specific field

    # test usage field
    assert result.usage is not None
    assert hasattr(result.usage, "input_tokens")
    assert hasattr(result.usage, "output_tokens")

    # test response structure for question-answer
    assert isinstance(result.response, list)
    assert len(result.response) == 2
    for answer in result.response:
        assert hasattr(answer, "question")
        assert hasattr(answer, "answer")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.parametrize("response_type", ("summary", "task"))
def test_lemur_get_response_data_additional_types(response_type, httpx_mock: HTTPXMock):
    """
    Tests whether additional LeMUR response types are correctly returned with request details.
    """
    request_id = "5678"

    # create a mock response - get_response_data returns LemurStringResponse but with different request details
    mock_lemur_response = factories.generate_dict_factory(
        factories.LemurStringResponse
    )()

    # Override the request details based on response type
    if response_type == "summary":
        mock_lemur_response["request"] = factories.generate_dict_factory(
            factories.LemurSummaryRequestDetails
        )()
    else:  # task
        mock_lemur_response["request"] = factories.generate_dict_factory(
            factories.LemurTaskRequestDetails
        )()

    mock_lemur_response["request_id"] = request_id

    # mock the specific endpoint
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{request_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_lemur_response,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur()
    result = lemur.get_response_data(request_id)

    # check the response type - get_response_data returns LemurStringResponse for all string responses
    assert isinstance(result, aai.LemurStringResponse)

    assert result.request_id == request_id

    # test the request field is populated correctly for all response types
    assert result.request is not None
    assert hasattr(result.request, "request_endpoint")
    assert hasattr(result.request, "temperature")
    assert hasattr(result.request, "final_model")
    assert hasattr(result.request, "max_output_size")
    assert hasattr(result.request, "created_at")

    # test type-specific request fields
    if response_type == "summary":
        assert result.request.request_endpoint == "/lemur/v3/summary"
        assert result.request.context is not None
        assert result.request.answer_format is not None
        assert result.request.prompt is None  # task-specific field
        assert result.request.questions is None  # qa-specific field
    else:  # task
        assert result.request.request_endpoint == "/lemur/v3/task"
        assert result.request.prompt is not None
        assert result.request.context is None  # summary-specific field
        assert result.request.answer_format is None  # summary-specific field
        assert result.request.questions is None  # qa-specific field

    # test usage field
    assert result.usage is not None
    assert hasattr(result.usage, "input_tokens")
    assert hasattr(result.usage, "output_tokens")

    # test string response field
    assert isinstance(result.response, str)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_get_response_data_with_optional_request_fields(httpx_mock: HTTPXMock):
    """
    Tests that optional fields in LemurRequestDetails are handled correctly.
    """
    request_id = "test-optional-fields"

    mock_lemur_response = factories.generate_dict_factory(
        factories.LemurStringResponse
    )()
    mock_lemur_response["request_id"] = request_id

    # Add optional fields to request details
    mock_lemur_response["request"]["transcript_ids"] = ["transcript_1", "transcript_2"]
    mock_lemur_response["request"]["input_text"] = "Test input text"
    mock_lemur_response["request"]["questions"] = [
        {"question": "What is this about?", "answer_format": "short"}
    ]
    mock_lemur_response["request"]["prompt"] = "Test prompt"
    mock_lemur_response["request"]["context"] = {"key": "value"}
    mock_lemur_response["request"]["answer_format"] = "bullet points"

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{request_id}",
        status_code=httpx.codes.OK,
        method="GET",
        json=mock_lemur_response,
    )

    lemur = aai.Lemur()
    result = lemur.get_response_data(request_id)

    assert isinstance(result, aai.LemurStringResponse)
    assert result.request_id == request_id

    # test that optional fields are present
    assert result.request.transcript_ids == ["transcript_1", "transcript_2"]
    assert result.request.input_text == "Test input text"
    assert result.request.questions is not None
    assert result.request.prompt == "Test prompt"
    assert result.request.context == {"key": "value"}
    assert result.request.answer_format == "bullet points"

    assert len(httpx_mock.get_requests()) == 1


def test_lemur_get_response_data_fails(httpx_mock: HTTPXMock):
    """
    Tests that get_response_data properly handles API errors.
    """
    request_id = "error-request-id"

    httpx_mock.add_response(
        url=f"{aai.settings.base_url}{ENDPOINT_LEMUR_BASE}/{request_id}",
        status_code=httpx.codes.NOT_FOUND,
        method="GET",
        json={"error": "Request not found"},
    )

    lemur = aai.Lemur()

    with pytest.raises(aai.LemurError):
        lemur.get_response_data(request_id)

    assert len(httpx_mock.get_requests()) == 1
