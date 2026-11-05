from pytest_httpx import HTTPXMock

import assemblyai as aai
import tests.unit.factories as factories
import tests.unit.unit_test_utils as test_utils

aai.settings.api_key = "test"


def test_action_items_request_model():
    req = aai.ActionItemsRequest(
        effort="low",
        include_decisions=True,
    )
    assert req.effort == "low"
    assert req.include_decisions is True


def test_action_items_request_defaults():
    req = aai.ActionItemsRequest()
    assert req.effort is None
    assert req.include_decisions is None


def test_action_items_in_speech_understanding_request():
    req = aai.SpeechUnderstandingRequest(
        request=aai.SpeechUnderstandingFeatureRequests(
            action_items=aai.ActionItemsRequest(
                effort="medium",
                include_decisions=False,
            )
        )
    )
    assert req.request is not None
    assert req.request.action_items is not None
    assert req.request.action_items.effort == "medium"
    assert req.request.action_items.include_decisions is False


def test_action_items_dict_style_in_transcription_config():
    config = aai.TranscriptionConfig(
        speech_understanding={
            "request": {
                "action_items": {
                    "effort": "low",
                    "include_decisions": True,
                }
            }
        }
    )
    assert config.speech_understanding is not None
    assert config.speech_understanding.request is not None
    assert config.speech_understanding.request.action_items is not None
    assert config.speech_understanding.request.action_items.effort == "low"
    assert config.speech_understanding.request.action_items.include_decisions is True


def test_action_items_response_model():
    resp = aai.ActionItemsResponse(
        status="success",
        effort="low",
        items=[
            aai.ActionItem(
                action_item="Send follow-up email",
                quote="I will send follow-up email",
                timestamp=124500,
            )
        ],
    )
    assert resp.status == "success"
    assert resp.effort == "low"
    assert resp.items is not None
    assert len(resp.items) == 1
    assert resp.items[0].action_item == "Send follow-up email"
    assert resp.items[0].quote == "I will send follow-up email"
    assert resp.items[0].timestamp == 124500


def test_transcript_speech_understanding_action_items(httpx_mock: HTTPXMock):
    mock_response = factories.generate_dict_factory(
        factories.TranscriptCompletedResponseFactory
    )()
    mock_response["speech_understanding"] = {
        "request": {
            "action_items": {
                "effort": "low",
                "include_decisions": True,
            }
        },
        "response": {
            "action_items": {
                "status": "success",
                "effort": "low",
                "items": [
                    {
                        "action_item": "Schedule sync meeting",
                        "quote": "Let's meet tomorrow at 10",
                        "timestamp": 45000,
                    }
                ],
            }
        },
    }

    request_body, transcript = test_utils.submit_mock_transcription_request(
        httpx_mock,
        mock_response,
        config=aai.TranscriptionConfig(
            speech_understanding={
                "request": {
                    "action_items": {
                        "effort": "low",
                        "include_decisions": True,
                    }
                }
            }
        ),
    )

    # Check request payload serialization
    assert request_body.get("speech_understanding") is not None
    su_request = request_body["speech_understanding"]["request"]
    assert su_request.get("action_items") is not None
    assert su_request["action_items"]["effort"] == "low"
    assert su_request["action_items"]["include_decisions"] is True

    # Check parsed transcript response
    assert transcript.speech_understanding is not None
    assert transcript.speech_understanding.response is not None
    assert transcript.speech_understanding.response.action_items is not None
    ai_resp = transcript.speech_understanding.response.action_items
    assert ai_resp.status == "success"
    assert ai_resp.effort == "low"
    assert ai_resp.items is not None
    assert len(ai_resp.items) == 1
    assert ai_resp.items[0].action_item == "Schedule sync meeting"
    assert ai_resp.items[0].quote == "Let's meet tomorrow at 10"
    assert ai_resp.items[0].timestamp == 45000
