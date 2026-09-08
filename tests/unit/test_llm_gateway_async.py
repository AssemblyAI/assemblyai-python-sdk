import json

import httpx
import pytest
from pytest_httpx import HTTPXMock, IteratorStream

import assemblyai as aai

pytestmark = pytest.mark.asyncio

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
                "global": {"completions": 15.0, "prompt": 3.0},
            },
            "creator": "anthropic",
            "retirement_date": 0,
            "available_regions": ["us", "eu"],
            "providers": ["anthropic"],
            "default_provider": "anthropic",
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


async def test_list_models_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked models endpoint
    _mock_ok(httpx_mock)

    # When listing models
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.models.list()

    # Then the response is parsed into an LLMGatewayModelList
    assert isinstance(result, aai.LLMGatewayModelList)
    assert result.data[0].id == "claude-sonnet-5"
    assert result.data[0].pricing.global_.completions == 15.0


async def test_models_requires_explicit_list_call(httpx_mock: HTTPXMock):
    # Given a mocked models endpoint
    _mock_ok(httpx_mock)

    # When calling .list() explicitly (gateway.models itself is not iterable)
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.models.list()

    # Then it returns the response envelope, matching openai-python's client.models.list()
    assert isinstance(result, aai.LLMGatewayModelList)
    assert [m.id for m in result.data] == ["claude-sonnet-5"]
    assert not hasattr(gateway.models, "__aiter__")


async def test_list_models_raises_llm_gateway_error_on_500(httpx_mock: HTTPXMock):
    # Given a mocked 500 response with the real business-error envelope
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
    async with aai.AsyncLLMGateway() as gateway:
        with pytest.raises(aai.LLMGatewayError) as exc_info:
            await gateway.models.list()

    # Then the error carries the server's status code, message, and request id
    error = exc_info.value
    assert error.status_code == httpx.codes.INTERNAL_SERVER_ERROR
    assert error.request_id == "req-abc-123"
    assert "something went wrong" in str(error)


async def test_error_handles_legacy_auth_error_envelope(httpx_mock: HTTPXMock):
    # Given a mocked 401 response using the legacy auth-middleware envelope
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
    async with aai.AsyncLLMGateway() as gateway:
        with pytest.raises(aai.LLMGatewayError) as exc_info:
            await gateway.models.list()

    # Then the legacy shape still resolves to a normal LLMGatewayError
    error = exc_info.value
    assert error.status_code == httpx.codes.UNAUTHORIZED
    assert error.request_id == "req-abc-125"


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
}


async def test_chat_completions_create_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked non-streaming completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

    # Then the response is parsed into an LLMGatewayChatCompletion
    assert isinstance(result, aai.LLMGatewayChatCompletion)
    assert result.choices[0].message.content == "Hello there."


async def test_chat_completions_stream_yields_chunks(httpx_mock: HTTPXMock):
    # Given a mocked streaming completions endpoint (OpenAI-shaped SSE)
    chunk = {
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
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        stream=IteratorStream(
            [f"data: {json.dumps(chunk)}\n\n".encode(), b"data: [DONE]\n\n"]
        ),
        headers={"content-type": "text/event-stream"},
    )

    # When creating a streaming completion
    async with aai.AsyncLLMGateway() as gateway:
        stream = await gateway.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [c async for c in stream]

    # Then it yields parsed chunks
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "Hello"


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


async def test_understanding_create_parses_response(httpx_mock: HTTPXMock):
    # Given a mocked understanding endpoint
    httpx_mock.add_response(
        url=UNDERSTANDING_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_UNDERSTANDING_RESPONSE,
    )

    # When running speaker identification against an existing transcript
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.understanding.create(
            transcript_id="transcript-123",
            request={"speaker_identification": {"speaker_type": "name"}},
        )

    # Then the response is parsed into an LLMGatewayUnderstandingResponse
    assert isinstance(result, aai.LLMGatewayUnderstandingResponse)
    assert result.utterances[0]["speaker"] == "John Doe"


async def test_understanding_validate_returns_none_on_success(httpx_mock: HTTPXMock):
    # Given a mocked validate endpoint returning an empty 200
    httpx_mock.add_response(
        url=UNDERSTANDING_VALIDATE_URL,
        method="POST",
        status_code=httpx.codes.OK,
    )

    # When validating a request
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.understanding.validate(
            request={"speaker_identification": {"speaker_type": "role"}}
        )

    # Then it returns None
    assert result is None


async def test_create_fallbacks_sent_as_given(httpx_mock: HTTPXMock):
    # Given a mocked completions endpoint
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    # When creating a completion with the API-expected fallbacks shape
    async with aai.AsyncLLMGateway() as gateway:
        await gateway.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hi"}],
            fallbacks=[
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
            ],
        )

    # Then it is forwarded into the request body untouched
    payload = json.loads(httpx_mock.get_requests()[0].read())
    assert payload["fallbacks"] == [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
    ]


_TOOL_CALL_RESPONSE = {
    "request_id": "req-tool-1",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
        }
    ],
    "usage": _COMPLETION_RESPONSE["usage"],
}


async def test_run_tools_completes_after_tool_call(httpx_mock: HTTPXMock):
    # Given a mocked completions endpoint that first requests a tool call, then answers
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_TOOL_CALL_RESPONSE,
    )
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_COMPLETION_RESPONSE,
    )

    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    calls = []

    def get_weather(city):
        calls.append(city)
        return {"forecast": "sunny"}

    # When running the tool loop
    async with aai.AsyncLLMGateway() as gateway:
        result = await gateway.chat.completions.run_tools(
            model="claude-sonnet-5",
            messages=messages,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            functions={"get_weather": get_weather},
        )

    # Then the tool was invoked with parsed arguments and the final completion is returned
    assert calls == ["Paris"]
    assert isinstance(result, aai.LLMGatewayChatCompletion)
    assert result.request_id == "req-completion-1"
    assert messages[-1] == {"role": "assistant", "content": "Hello there."}


async def test_run_tools_raises_on_unknown_tool(httpx_mock: HTTPXMock):
    # Given a mocked response requesting an unregistered tool
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        json=_TOOL_CALL_RESPONSE,
    )

    # When running the tool loop with no matching function
    async with aai.AsyncLLMGateway() as gateway:
        with pytest.raises(ValueError, match="get_weather"):
            await gateway.chat.completions.run_tools(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                functions={},
            )


async def test_run_tools_raises_after_max_rounds(httpx_mock: HTTPXMock):
    # Given a mock that always requests the same tool call
    for _ in range(2):
        httpx_mock.add_response(
            url=COMPLETIONS_URL,
            method="POST",
            status_code=httpx.codes.OK,
            json=_TOOL_CALL_RESPONSE,
        )

    # When running the tool loop with a max_rounds it can't finish within
    async with aai.AsyncLLMGateway() as gateway:
        with pytest.raises(RuntimeError, match="max_rounds"):
            await gateway.chat.completions.run_tools(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                functions={"get_weather": lambda city: "sunny"},
                max_rounds=2,
            )


async def test_run_tools_rejects_streaming():
    # When running the tool loop with stream=True
    async with aai.AsyncLLMGateway() as gateway:
        with pytest.raises(ValueError, match="stream"):
            await gateway.chat.completions.run_tools(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                functions={},
                stream=True,
            )


async def test_chat_completions_stream_get_final_message(httpx_mock: HTTPXMock):
    # Given a mocked streaming completions endpoint
    chunk1 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "claude-sonnet-5",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "claude-sonnet-5",
        "choices": [
            {"index": 0, "delta": {"content": " there."}, "finish_reason": "stop"}
        ],
        "usage": _COMPLETION_RESPONSE["usage"],
    }
    httpx_mock.add_response(
        url=COMPLETIONS_URL,
        method="POST",
        status_code=httpx.codes.OK,
        stream=IteratorStream(
            [
                f"data: {json.dumps(chunk1)}\n\n".encode(),
                f"data: {json.dumps(chunk2)}\n\n".encode(),
                b"data: [DONE]\n\n",
            ]
        ),
        headers={"content-type": "text/event-stream"},
    )

    # When consuming the stream and asking for the final assembled message
    async with aai.AsyncLLMGateway() as gateway:
        stream = await gateway.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [c async for c in stream]
        final = stream.get_final_message()

    # Then it still yields the raw chunks, and the accumulator has the assembled result
    assert len(chunks) == 2
    assert final.content == "Hello there."
    assert final.finish_reason == "stop"
    assert final.usage.total_tokens == 15


async def test_aclose_closes_owned_client_only():
    # Given a gateway with its own client, and one sharing a caller-owned client
    async with aai.AsyncLLMGateway() as owned:
        pass
    assert owned.client.http_client.is_closed

    async with aai.AsyncClient(settings=aai.settings) as shared_client:
        async with aai.AsyncLLMGateway(client=shared_client) as gateway:
            pass
        # Then a caller-supplied client is left open for the caller to close
        assert not shared_client.http_client.is_closed
