import uuid

import httpx
import pytest
from pytest_httpx import HTTPXMock

import assemblyai as aai
from tests.unit import factories

aai.settings.api_key = "test"


def test_lemur_single_question_succeeds(httpx_mock: HTTPXMock):
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
        url=f"{aai.settings.base_url}/generate/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )
    answer = lemur.question(question)

    # check whether answer is not a list
    assert isinstance(answer, aai.LemurQuestionResult)

    # check the response
    assert answer.question == mock_lemur_answer["response"][0]["question"]
    assert answer.answer == mock_lemur_answer["response"][0]["answer"]

    # check whether it is using the default model
    assert mock_lemur_answer["model"] == aai.LemurModel.default

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_multiple_question_succeeds(httpx_mock: HTTPXMock):
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
        url=f"{aai.settings.base_url}/generate/question-answer",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_answer,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )
    answers = lemur.question(questions=questions)

    # check whether answers is a list
    assert isinstance(answers, list)

    # check the response
    for idx, answer in enumerate(answers):
        assert answer.question == mock_lemur_answer["response"][idx]["question"]
        assert answer.answer == mock_lemur_answer["response"][idx]["answer"]

    # check whether it is using the default model
    assert mock_lemur_answer["model"] == aai.LemurModel.default

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
        url=f"{aai.settings.base_url}/generate/question-answer",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )

    with pytest.raises(aai.LemurError):
        lemur.question(question)

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR succeeds.
    """

    # create a mock response of a LemurSummaryResponse
    mock_lemur_summary = factories.generate_dict_factory(
        factories.LemurSummaryResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/generate/summary",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_summary,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )
    summary = lemur.summarize(context="Callers asking for cars", answer_format="TLDR")

    # check the response
    assert summary == mock_lemur_summary["response"]

    # check whether it is using the default model
    assert mock_lemur_summary["model"] == aai.LemurModel.default

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_summarize_fails(httpx_mock: HTTPXMock):
    """
    Tests whether summarizing a transcript via LeMUR fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/generate/summary",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )

    with pytest.raises(aai.LemurError):
        lemur.summarize(context="Callers asking for cars", answer_format="TLDR")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_ask_coach_succeeds(httpx_mock: HTTPXMock):
    """
    Tests whether asking a coach question succeeds.
    """

    # create a mock response of a LemurSummaryResponse
    mock_lemur_ask_coach = factories.generate_dict_factory(
        factories.LemurSummaryResponse
    )()

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/generate/ai-coach",
        status_code=httpx.codes.OK,
        method="POST",
        json=mock_lemur_ask_coach,
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )
    feedback = lemur.ask_coach(context="Callers asking for cars")

    # check the response
    assert feedback == mock_lemur_ask_coach["response"]

    # check whether it is using the default model
    assert mock_lemur_ask_coach["model"] == aai.LemurModel.default

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1


def test_lemur_ask_coach_fails(httpx_mock: HTTPXMock):
    """
    Tests whether asking a coach question fails.
    """

    # mock the specific endpoints
    httpx_mock.add_response(
        url=f"{aai.settings.base_url}/generate/ai-coach",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        method="POST",
        json={"error": "something went wrong"},
    )

    # mimic the usage of the SDK
    lemur = aai.Lemur(
        transcript_ids=[
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
    )

    with pytest.raises(aai.LemurError):
        lemur.ask_coach(context="Callers asking for cars")

    # check whether we mocked everything
    assert len(httpx_mock.get_requests()) == 1
