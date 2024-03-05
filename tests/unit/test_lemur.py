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

    # create a mock response of a LemurSummaryResponse
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
    result = lemur.task(prompt="Create action items of the meeting")

    # check the response
    assert isinstance(result, aai.LemurTaskResponse)

    assert result.response == mock_lemur_task_response["response"]

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_task_succeeds_input_text(httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds.
    """

    # create a mock response of a LemurSummaryResponse
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
        aai.LemurModel.mistral7b,
        aai.LemurModel.claude2_1,
    ),
)
def test_lemur_task_succeeds(final_model, httpx_mock: HTTPXMock):
    """
    Tests whether creating a task request succeeds with other models.
    """

    # create a mock response of a LemurSummaryResponse
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

    # create a mock response of a LemurSummaryResponse
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
