import json

import pytest
from pytest_httpx import HTTPXMock, IteratorStream

import assemblyai as aai


def _response(data_prefix, done_prefix):
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }
    # A following frame makes ignoring [DONE] observable, rather than relying
    # on the response ending immediately after the sentinel.
    after_done = {
        **chunk,
        "choices": [{"index": 0, "delta": {"content": "after done"}}],
    }
    return IteratorStream(
        [
            b": keepalive\r\nevent: message\r\n\r\ndata:\r\n\r\n",
            (data_prefix + json.dumps(chunk) + "\r\n\r\n").encode(),
            (done_prefix + "[DONE]\r\n\r\n").encode(),
            ("data: " + json.dumps(after_done) + "\r\n\r\n").encode(),
        ]
    )


@pytest.mark.parametrize("data_prefix", ["data:", "data: "])
@pytest.mark.parametrize("done_prefix", ["data:", "data: "])
def test_stream_accepts_optional_data_space(
    httpx_mock: HTTPXMock, data_prefix, done_prefix
):
    client = aai.Client(api_key="test")
    try:
        httpx_mock.add_response(
            method="POST",
            url=client.settings.llm_gateway_base_url + "/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=_response(data_prefix, done_prefix),
        )
        chunks = list(
            aai.LLMGateway(client=client).chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )
        assert [chunk.choices[0].delta.content for chunk in chunks] == ["Hello"]
        assert client.last_response.is_closed
    finally:
        client.http_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_prefix", ["data:", "data: "])
@pytest.mark.parametrize("done_prefix", ["data:", "data: "])
async def test_async_stream_accepts_optional_data_space(
    httpx_mock: HTTPXMock, data_prefix, done_prefix
):
    async with aai.AsyncClient(api_key="test") as client:
        httpx_mock.add_response(
            method="POST",
            url=client.settings.llm_gateway_base_url + "/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            stream=_response(data_prefix, done_prefix),
        )
        stream = await aai.AsyncLLMGateway(client=client).chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        assert [chunk.choices[0].delta.content for chunk in chunks] == ["Hello"]
        assert client.last_response.is_closed
