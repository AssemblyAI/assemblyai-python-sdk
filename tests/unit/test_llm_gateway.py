import httpx
import pytest
from pytest_httpx import HTTPXMock, IteratorStream

import assemblyai as aai
from assemblyai.llm_gateway.v1 import models

aai.settings.api_key = "test"

MODELS_URL = f"{aai.settings.llm_gateway_base_url}/v1/models"
COMPLETIONS_URL = f"{aai.settings.llm_gateway_base_url}/v1/chat/completions"
UNDERSTANDING_URL = f"{aai.settings.llm_gateway_base_url}/v1/understanding"
UNDERSTANDING_VALIDATE_URL = (
    f"{aai.settings.llm_gateway_base_url}/v1/understanding/validate"
)

_OK_RESPONSE = {
    "data": [
        {
            "id": "claude-sonnet-5",
            "name": "Claude Sonnet 5",
            "description": "Anthropic's Claude Sonnet 5",
            "default_parameters": {
                "temperature": 1.0,
                "top_p": 0.9,
                "frequency_penalty": 0,
            },
            "supported_parameters": ["temperature", "top_p", "tools"],
            "top_provider": {
                "is_moderated": False,
                "context_length": 200000,
                "max_completion_tokens": 8192,
            },
            "context_length": 200000,
            "pricing": {
                "us": {"completions": 15.0, "prompt": 3.0},
                "eu": {"completions": 16.5, "prompt": 3.3},
                "global": {"completions": 15.0, "prompt": 3.0},
                "regional_increase_percent": 10.0,
            },
            "creator": "anthropic",
            "retirement_date": 0,
            "available_regions": ["us", "eu"],
            "providers": [{"id": "anthropic", "name": "Anthropic"}],
            "default_provider": {"id": "anthropic", "name": "Anthropic"},
        },
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "description": "OpenAI's GPT-4o",
            "default_parameters": {
                "temperature": None,
                "top_p": None,
                "frequency_penalty": None,
            },
            "supported_parameters": ["temperature"],
            "top_provider": {
                "is_moderated": True,
                "context_length": 128000,
                "max_completion_tokens": 4096,
            },
            "context_length": 128000,
            "pricing": {
                "global": {"completions": 10.0, "prompt": 2.5},
            },
            "creator": "openai",
            "retirement_date": 0,
            "available_regions": ["global"],
            "providers": [{"id": "openai", "name": "Open AI"}],
            "default_provider": {"id": "openai", "name": "Open AI"},
        },
    ]
}


def _mock_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.OK,
        json=_OK_RESPONSE,
    )


def test_list_models_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked models endpoint
    _mock_ok(httpx_mock)

    # When listing models
    result = aai.LLMGateway().models.list()

    # Then the response is parsed into an LLMGatewayModelList
    assert isinstance(result, aai.LLMGatewayModelList)
    assert len(result.data) == 2
    first = result.data[0]
    assert first.id == "claude-sonnet-5"
    assert first.default_parameters.temperature == 1.0
    assert first.pricing.us.completions == 15.0
    assert first.pricing.eu.prompt == 3.3
    assert first.pricing.global_.completions == 15.0
    assert first.pricing.regional_increase_percent == 10.0


def test_list_models_handles_minimal_payload(httpx_mock: HTTPXMock):
    # Given a mocked endpoint with all optional fields omitted
    _mock_ok(httpx_mock)

    # When listing models
    result = aai.LLMGateway().models.list()

    # Then omitted optional fields default to None
    second = result.data[1]
    assert second.pricing.us is None
    assert second.pricing.eu is None
    assert second.pricing.regional_increase_percent is None
    assert second.default_parameters.temperature is None


def test_models_requires_explicit_list_call(httpx_mock: HTTPXMock):
    # Given a mocked models endpoint
    _mock_ok(httpx_mock)

    # When calling .list() explicitly (gateway.models itself is not iterable)
    result = aai.LLMGateway().models.list()

    # Then it returns the response envelope, matching openai-python's client.models.list()
    assert isinstance(result, aai.LLMGatewayModelList)
    ids = [m.id for m in result.data]
    assert ids == ["claude-sonnet-5", "gpt-4o"]
    assert not hasattr(aai.LLMGateway().models, "__iter__")


def test_gateway_uses_llm_gateway_base_url(httpx_mock: HTTPXMock):
    # Given a mocked models endpoint
    _mock_ok(httpx_mock)

    # When listing models
    aai.LLMGateway().models.list()

    # Then the request targets llm_gateway_base_url, not the generic base_url
    request = httpx_mock.get_requests()[0]
    assert str(request.url) == MODELS_URL
    assert aai.settings.llm_gateway_base_url != aai.settings.base_url


def test_list_models_raises_llm_gateway_error_on_500(httpx_mock: HTTPXMock):
    # Given a mocked 500 response with the real business-error envelope
    # ({"message", "code", "request_id", "metadata": {"errors": [...]}})
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        json={
            "message": "something went wrong",
            "code": 500,
            "request_id": "req-abc-123",
        },
    )

    # When listing models
    with pytest.raises(aai.LLMGatewayError) as exc_info:
        aai.LLMGateway().models.list()

    # Then the error carries the server's status code, message, and request id
    error = exc_info.value
    assert error.status_code == httpx.codes.INTERNAL_SERVER_ERROR
    assert error.request_id == "req-abc-123"
    assert error.errors is None
    assert "something went wrong" in str(error)


def test_error_carries_metadata_errors_list(httpx_mock: HTTPXMock):
    # Given a mocked 400 response with metadata.errors populated
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.BAD_REQUEST,
        json={
            "message": "invalid request body",
            "code": 400,
            "request_id": "req-abc-124",
            "metadata": {"errors": ["target_languages is empty"]},
        },
    )

    # When listing models
    with pytest.raises(aai.LLMGatewayError) as exc_info:
        aai.LLMGateway().models.list()

    # Then .errors carries the validation messages
    assert exc_info.value.errors == ["target_languages is empty"]


def test_error_handles_legacy_auth_error_envelope(httpx_mock: HTTPXMock):
    # Given a mocked 401 response using the legacy auth-middleware envelope
    # ({"error", "status", "request_id"} — raised before any route handler runs)
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.UNAUTHORIZED,
        json={
            "error": "Authentication error, API token missing/invalid",
            "status": "error",
            "request_id": "req-abc-125",
        },
    )

    # When listing models
    with pytest.raises(aai.LLMGatewayError) as exc_info:
        aai.LLMGateway().models.list()

    # Then the legacy shape still resolves to a normal LLMGatewayError
    error = exc_info.value
    assert error.status_code == httpx.codes.UNAUTHORIZED
    assert error.request_id == "req-abc-125"
    assert "Authentication error" in str(error)


def test_list_models_raises_on_malformed_error_body(httpx_mock: HTTPXMock):
    # Given a mocked 500 response with a non-JSON body
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        content=b"upstream timeout",
    )

    # When listing models
    with pytest.raises(aai.LLMGatewayError) as exc_info:
        aai.LLMGateway().models.list()

    # Then a fallback message is still raised
    error = exc_info.value
    assert error.status_code == httpx.codes.INTERNAL_SERVER_ERROR
    assert "upstream timeout" in str(error)


_COMPLETION_RESPONSE = {
    "request_id": "req-completion-1",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello there."},
        }
    ],
    "usage": {
        "input_tokens": 10,
        "prompt_tokens": 10,
        "output_tokens": 5,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    "http_status_code": 200,
    "response_time": 123456789,
    "llm_status_code": 200,
}


def test_chat_completions_create_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked non-streaming completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion
    result = aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Then the response is parsed into an LLMGatewayChatCompletion
    assert isinstance(result, aai.LLMGatewayChatCompletion)
    assert result.request_id == "req-completion-1"
    assert result.choices[0].message.content == "Hello there."
    assert result.usage.total_tokens == 15
    assert result.response_time == 123456789


def test_chat_completions_create_sends_extension_kwargs(httpx_mock: HTTPXMock):
    # Given a mocked completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion with AssemblyAI extension kwargs
    aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
        fallbacks=[
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        ],
        zero_data_retention=True,
        transcript_id="transcript-123",
    )

    # Then the extension fields land in the posted JSON body untouched
    body = httpx_mock.get_requests()[0].read()
    import json as _json

    payload = _json.loads(body)
    assert payload["zero_data_retention"] is True
    assert payload["transcript_id"] == "transcript-123"
    assert payload["fallbacks"][0]["model"] == "gpt-4o"
    assert payload["stream"] is False


def test_chat_completions_stream_yields_chunks(httpx_mock: HTTPXMock):
    # Given a mocked streaming completions endpoint (OpenAI-shaped SSE)
    chunk1 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "claude-sonnet-5",
        "service_tier": None,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        "usage": None,
        "obfuscation": "",
    }
    chunk2 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "claude-sonnet-5",
        "service_tier": None,
        "system_fingerprint": None,
        "choices": [
            {"index": 0, "delta": {"content": " there."}, "finish_reason": "stop"}
        ],
        "usage": None,
        "obfuscation": "",
    }
    import json as _json

    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        stream=IteratorStream(
            [
                f"data: {_json.dumps(chunk1)}\n\n".encode(),
                f"data: {_json.dumps(chunk2)}\n\n".encode(),
                b"data: [DONE]\n\n",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )

    # When creating a streaming completion
    chunks = list(
        aai.LLMGateway().chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
    )

    # Then it yields parsed chunks and stops at [DONE]
    assert len(chunks) == 2
    assert all(isinstance(c, aai.LLMGatewayCompletionChunk) for c in chunks)
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].finish_reason == "stop"


def test_chat_completions_stream_raises_on_setup_error(httpx_mock: HTTPXMock):
    # Given a mocked 500 response before any streaming bytes are sent
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.INTERNAL_SERVER_ERROR,
        json={"message": "something went wrong", "code": 500, "request_id": "req-x"},
    )

    # When creating a streaming completion and consuming the iterator
    with pytest.raises(aai.LLMGatewayError):
        list(
            aai.LLMGateway().chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )


_UNDERSTANDING_RESPONSE = {
    "speech_understanding": {
        "request": {"speaker_identification": {"speaker_type": "name"}},
        "response": {
            "speaker_identification": {
                "status": "success",
                "mapping": {"A": "John Doe"},
            }
        },
    },
    "utterances": [{"speaker": "John Doe", "text": "Hello"}],
    "request_id": "req-understanding-1",
}


def test_understanding_create_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked understanding endpoint
    httpx_mock.add_response(
        url=UNDERSTANDING_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_UNDERSTANDING_RESPONSE,
    )

    # When running speaker identification against an existing transcript
    result = aai.LLMGateway().understanding.create(
        transcript_id="transcript-123",
        request={"speaker_identification": {"speaker_type": "name"}},
    )

    # Then the response is parsed into an LLMGatewayUnderstandingResponse
    assert isinstance(result, aai.LLMGatewayUnderstandingResponse)
    assert result.request_id == "req-understanding-1"
    assert result.utterances[0]["speaker"] == "John Doe"
    assert result.speech_understanding["response"]["speaker_identification"][
        "mapping"
    ] == {"A": "John Doe"}


def test_understanding_create_sends_transcript_id_and_request(httpx_mock: HTTPXMock):
    # Given a mocked understanding endpoint
    httpx_mock.add_response(
        url=UNDERSTANDING_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_UNDERSTANDING_RESPONSE,
    )

    # When calling create()
    aai.LLMGateway().understanding.create(
        transcript_id="transcript-123",
        request={"speaker_identification": {"speaker_type": "name", "effort": "low"}},
    )

    # Then the posted body carries transcript_id and the nested feature request
    import json as _json

    payload = _json.loads(httpx_mock.get_requests()[0].read())
    assert payload["transcript_id"] == "transcript-123"
    assert (
        payload["speech_understanding"]["request"]["speaker_identification"]["effort"]
        == "low"
    )


def test_understanding_validate_returns_none_on_success(httpx_mock: HTTPXMock):
    # Given a mocked validate endpoint returning an empty 200
    httpx_mock.add_response(
        url=UNDERSTANDING_VALIDATE_URL,
        method="POST",
        status_code=httpx.codes.OK,
    )

    # When validating a request
    result = aai.LLMGateway().understanding.validate(
        request={"speaker_identification": {"speaker_type": "role"}}
    )

    # Then it returns None
    assert result is None


def test_understanding_validate_raises_with_errors_list(httpx_mock: HTTPXMock):
    # Given a mocked validate endpoint returning a 400 with validation messages
    httpx_mock.add_response(
        url=UNDERSTANDING_VALIDATE_URL,
        method="POST",
        status_code=httpx.codes.BAD_REQUEST,
        json={
            "message": "invalid request body",
            "code": 400,
            "request_id": "req-validate-1",
            "metadata": {"errors": ["target_languages is empty"]},
        },
    )

    # When validating an invalid request
    with pytest.raises(aai.LLMGatewayError) as exc_info:
        aai.LLMGateway().understanding.validate(request={"translation": {}})

    # Then .errors carries the validation messages
    assert exc_info.value.errors == ["target_languages is empty"]


def test_create_fallbacks_sent_as_given(httpx_mock: HTTPXMock):
    # Given a mocked completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion with the API-expected fallbacks shape
    aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
        fallbacks=[
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        ],
    )

    # Then it is forwarded into the request body untouched
    import json as _json

    payload = _json.loads(httpx_mock.get_requests()[0].read())
    assert payload["fallbacks"] == [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
    ]


def test_create_without_fallbacks_omits_field(httpx_mock: HTTPXMock):
    # Given a mocked completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion without fallbacks
    aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Then the field is omitted entirely, not sent as null
    import json as _json

    payload = _json.loads(httpx_mock.get_requests()[0].read())
    assert "fallbacks" not in payload


# The gateway is written in Go, where a nil slice/map marshals as `null` rather
# than `[]`/`{}`. An explicit null bypasses a pydantic field default, so each
# collection on the response path has to tolerate it.


def test_list_models_tolerates_null_data(httpx_mock: HTTPXMock):
    # Given a models response whose `data` slice came back null
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.OK,
        json={"data": None},
    )

    # When listing models
    result = aai.LLMGateway().models.list()

    # Then it parses to an empty list rather than raising
    assert result.data == []


def test_list_models_tolerates_null_model_collections(httpx_mock: HTTPXMock):
    # Given a model entry whose list fields all came back null
    httpx_mock.add_response(
        url=MODELS_URL,
        method="GET",
        status_code=httpx.codes.OK,
        json={
            "data": [
                {
                    "id": "claude-sonnet-5",
                    "name": "Claude Sonnet 5",
                    "description": "",
                    "default_parameters": {},
                    "supported_parameters": None,
                    "top_provider": {
                        "is_moderated": False,
                        "context_length": 200000,
                        "max_completion_tokens": 8192,
                    },
                    "context_length": 200000,
                    "pricing": {"global": {"completions": 15.0, "prompt": 3.0}},
                    "creator": "anthropic",
                    "retirement_date": 0,
                    "available_regions": None,
                    "providers": None,
                    "default_provider": {"id": "bedrock", "name": "AWS Bedrock"},
                }
            ]
        },
    )

    # When listing models
    model = aai.LLMGateway().models.list().data[0]

    # Then every null collection defaults to empty
    assert model.supported_parameters == []
    assert model.available_regions == []
    assert model.providers == []


def test_chat_completions_tolerates_null_choices(httpx_mock: HTTPXMock):
    # Given a completion whose `choices` slice came back null
    response = dict(_COMPLETION_RESPONSE, choices=None)
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When creating a completion
    result = aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Then it parses to an empty list rather than raising
    assert result.choices == []


def test_chat_completions_tolerates_usage_without_token_details(
    httpx_mock: HTTPXMock,
):
    # Given a completion whose usage omits the OpenAI-shaped detail objects
    response = dict(
        _COMPLETION_RESPONSE,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When creating a completion
    result = aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Then the totals still parse and the detail objects are None
    assert result.usage.total_tokens == 15
    assert result.usage.prompt_tokens_details is None
    assert result.usage.completion_tokens_details is None


def test_understanding_tolerates_null_speech_understanding(httpx_mock: HTTPXMock):
    # Given an understanding response whose `speech_understanding` map came back null
    httpx_mock.add_response(
        url=UNDERSTANDING_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json={"speech_understanding": None, "request_id": "req-1"},
    )

    # When running understanding
    result = aai.LLMGateway().understanding.create(
        transcript_id="transcript-1",
        request={"translation": {"target_languages": ["es"]}},
    )

    # Then it parses to an empty dict rather than raising
    assert result.speech_understanding == {}


def test_chat_completions_preserves_unknown_server_fields(httpx_mock: HTTPXMock):
    # Given a completion carrying a field this SDK version doesn't model
    response = dict(_COMPLETION_RESPONSE, brand_new_field={"nested": 1})
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=response,
    )

    # When creating a completion
    result = aai.LLMGateway().chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # Then the unknown field is still reachable, not dropped
    assert result.brand_new_field == {"nested": 1}


def test_stream_chunk_exposes_tool_call_delta(httpx_mock: HTTPXMock):
    # Given a Claude-shaped streaming chunk carrying a fragmented tool call
    # (see the gateway's pkg/models/claude/stream.go Delta/OpenAIToolCalls)
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "claude-haiku-4-5-20251001",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"ci',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    import json as _json

    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        stream=IteratorStream(
            [f"data: {_json.dumps(chunk)}\n\n".encode(), b"data: [DONE]\n\n"]
        ),
        headers={"content-type": "text/event-stream"},
    )

    # When consuming the stream
    chunks = list(
        aai.LLMGateway().chat.completions.create(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            stream=True,
        )
    )

    # Then the delta's role and tool-call fragment are typed and reachable
    delta = chunks[0].choices[0].delta
    assert delta.role == "assistant"
    assert delta.content is None
    tool_call = delta.tool_calls[0]
    assert isinstance(tool_call, models.LLMGatewayChunkToolCall)
    assert tool_call.index == 0
    assert tool_call.id == "call_1"
    assert tool_call.function.name == "get_weather"
    # arguments stream in fragments; the SDK doesn't reassemble them
    assert tool_call.function.arguments == '{"ci'
