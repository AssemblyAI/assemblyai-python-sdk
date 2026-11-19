import asyncio
from contextlib import asynccontextmanager

import pytest
from websockets.legacy.server import serve

from assemblyai.streaming.v3 import (
    AsyncRealTimeTranscriber,
    RealTimeParameters,
    RealTimeTranscriberOptions,
    SpeechModel,
)

pytestmark = pytest.mark.asyncio


def params():
    return RealTimeParameters(
        sample_rate=16000,
        speech_model=SpeechModel.universal_streaming_english,
    )


@asynccontextmanager
async def delayed_server():
    """Real sockets; only the HTTP upgrade response is held."""
    entered = asyncio.Event()
    release = asyncio.Event()
    connections = []
    requests = []

    async def handshake(path, headers):
        requests.append(path)
        entered.set()
        await release.wait()

    async def handler(socket):
        connections.append(socket)
        await socket.wait_closed()

    before = asyncio.all_tasks()
    async with serve(
        handler, "127.0.0.1", 0, process_request=handshake, close_timeout=0.1
    ) as server:
        port = server.sockets[0].getsockname()[1]
        client = AsyncRealTimeTranscriber(
            RealTimeTranscriberOptions(
                api_key="local-test", api_host=f"ws://127.0.0.1:{port}"
            )
        )
        try:
            yield client, entered, release, connections, requests
        finally:
            release.set()
            await client.disconnect()
            # Baseline can orphan both SDK tasks and sockets. Clean the test
            # fixture explicitly, after measuring SDK-owned cleanup.
            for task in asyncio.all_tasks() - before:
                if task.get_name().startswith("AsyncRealTimeTranscriber."):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            for socket in connections:
                await socket.close()


@pytest.mark.parametrize("terminate", [False, True])
async def test_disconnect_interrupts_pending_handshake(terminate):
    async with delayed_server() as (client, entered, release, sockets, requests):
        pending = asyncio.create_task(client.connect(params()))
        await asyncio.wait_for(entered.wait(), 0.5)
        await client.disconnect(terminate=terminate)
        stopped_on_return = pending.done()
        release.set()
        result = (await asyncio.gather(pending, return_exceptions=True))[0]
        print(
            f"terminate={terminate} stopped_on_return={stopped_on_return} "
            f"result={type(result).__name__} "
            f"open_sockets={sum(s.close_code is None for s in sockets)}"
        )
        assert stopped_on_return
        assert isinstance(result, asyncio.CancelledError)
        assert all(socket.close_code is not None for socket in sockets)
        await client.disconnect(terminate=terminate)
        with pytest.raises(RuntimeError, match="already been connected"):
            await client.connect(params())
        assert len(requests) == 1


async def test_concurrent_connect_reserves_single_use_before_handshake():
    async with delayed_server() as (client, entered, release, sockets, requests):
        first = asyncio.create_task(client.connect(params()))
        await asyncio.wait_for(entered.wait(), 0.5)
        second = asyncio.create_task(client.connect(params()))
        # Release only after the second coroutine has had an opportunity to
        # run its single-use check, without waiting for a second handshake.
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        print(f"connect_results={[type(r).__name__ for r in results]}")
        assert results[0] is None
        assert isinstance(results[1], RuntimeError)
        assert len(requests) == len(sockets) == 1
        await client.disconnect()
        await asyncio.wait_for(sockets[0].wait_closed(), 0.5)


async def test_cancelled_handshake_remains_single_use():
    async with delayed_server() as (client, entered, release, sockets, requests):
        pending = asyncio.create_task(client.connect(params()))
        await asyncio.wait_for(entered.wait(), 0.5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        with pytest.raises(RuntimeError, match="already been connected"):
            await client.connect(params())
        await client.disconnect()
        assert len(requests) == 1


async def test_normal_connect_and_repeated_disconnect():
    async with delayed_server() as (client, entered, release, sockets, requests):
        release.set()
        await client.connect(params())
        with pytest.raises(RuntimeError, match="already been connected"):
            await client.connect(params())
        await client.stream(b"\0" * 320)
        await client.disconnect()
        await client.disconnect()
        await asyncio.wait_for(sockets[0].wait_closed(), 0.5)
        assert len(requests) == 1
