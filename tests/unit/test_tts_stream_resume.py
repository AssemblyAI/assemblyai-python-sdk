"""Public AsyncTextToSpeech streamed-body ownership regressions over real HTTP."""

import asyncio
from contextlib import asynccontextmanager

import pytest

import assemblyai as aai

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def speech_server():
    release = asyncio.Event()
    tasks = set()
    requests = []
    disconnected = asyncio.Event()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            headers = dict(
                line.lower().split(": ", 1)
                for line in header.decode().split("\r\n")[1:]
                if ": " in line
            )
            requests.append(await reader.readexactly(int(headers["content-length"])))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n"
                b"X-Request-Id: local-speech\r\n\r\n"
                b"4\r\nhead\r\n"
            )
            await writer.drain()
            # A real chunk boundary, not arbitrary TCP segmentation. The
            # client receives the first bytes while generation is unfinished.
            releasing = asyncio.create_task(release.wait())
            closing = asyncio.create_task(reader.read(1))
            try:
                done, _ = await asyncio.wait(
                    {releasing, closing}, return_when=asyncio.FIRST_COMPLETED
                )
                if closing in done:
                    disconnected.set()
                    return
            finally:
                releasing.cancel()
                closing.cancel()
                await asyncio.gather(releasing, closing, return_exceptions=True)
            writer.write(b"4\r\ntail\r\n0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", release, requests, disconnected
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


@pytest.mark.parametrize("resume", ["read", "iterator"])
async def test_partial_read_can_resume_without_replaying_or_reposting(resume):
    async with speech_server() as (url, release, requests, disconnected):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", tts_base_url=url)
        ) as client:
            tts = aai.AsyncTextToSpeech(client=client)
            async with tts.stream("Hello.") as speech:
                iterator = speech.aiter_bytes()
                assert await iterator.__anext__() == b"head"
                await iterator.aclose()  # release this wrapper, not the speech
                release.set()
                if resume == "read":
                    remaining = await speech.aread()
                else:
                    remaining = b"".join([chunk async for chunk in speech])
                print(
                    f"resume={resume} remaining={remaining!r} requests={len(requests)}"
                )
                assert remaining == b"tail"
                assert await speech.aread() == b""
                assert len(requests) == 1
                assert speech.info.request_id == "local-speech"
            await speech.aclose()


async def test_two_partial_streams_share_pool_without_sharing_body_cursor():
    async with speech_server() as (url, release, requests, disconnected):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", tts_base_url=url)
        ) as client:
            tts = aai.AsyncTextToSpeech(client=client)
            async with tts.stream("One.") as one, tts.stream("Two.") as two:
                it1, it2 = one.aiter_bytes(), two.aiter_bytes()
                assert await asyncio.gather(it1.__anext__(), it2.__anext__()) == [
                    b"head",
                    b"head",
                ]
                await it1.aclose()
                await it2.aclose()
                release.set()
                assert await asyncio.gather(one.aread(), two.aread()) == [
                    b"tail",
                    b"tail",
                ]
                assert len(requests) == 2


async def test_same_iterator_still_reads_all_chunks():
    async with speech_server() as (url, release, requests, disconnected):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", tts_base_url=url)
        ) as client:
            async with aai.AsyncTextToSpeech(client=client).stream("Hello.") as speech:
                iterator = speech.aiter_bytes()
                assert await iterator.__anext__() == b"head"
                release.set()
                assert b"".join([chunk async for chunk in iterator]) == b"tail"
                assert len(requests) == 1


async def test_cancelled_read_releases_partial_body_on_context_exit():
    async with speech_server() as (url, release, requests, disconnected):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", tts_base_url=url)
        ) as client:
            speech = aai.AsyncTextToSpeech(client=client).stream("Hello.")
            async with speech:
                iterator = speech.aiter_bytes()
                assert await iterator.__anext__() == b"head"
                reading = asyncio.create_task(iterator.__anext__())
                await asyncio.sleep(0.02)
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
                assert reading.cancelled()
            await speech.aclose()
            await asyncio.wait_for(disconnected.wait(), 2)
            assert len(requests) == 1


@pytest.mark.parametrize("resume", ["read", "iterator"])
async def test_sync_partial_read_can_resume(resume):
    async with speech_server() as (url, release, requests, disconnected):
        loop = asyncio.get_running_loop()

        def consume():
            client = aai.Client(settings=aai.Settings(api_key="test", tts_base_url=url))
            try:
                with aai.TextToSpeech(client=client).stream("Hello.") as speech:
                    iterator = speech.iter_bytes()
                    assert next(iterator) == b"head"
                    iterator.close()
                    loop.call_soon_threadsafe(release.set)
                    remaining = speech.read() if resume == "read" else b"".join(speech)
                    print(f"sync resume={resume} remaining={remaining!r}")
                    assert remaining == b"tail"
                    assert speech.read() == b""
                speech.close()
            finally:
                client.close()

        await asyncio.wait_for(loop.run_in_executor(None, consume), 4)
        assert len(requests) == 1
