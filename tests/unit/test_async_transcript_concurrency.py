"""Real HTTP regression: a delayed old poll must not erase a finished result."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

import assemblyai as aai

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def poll_server(terminal, old_status="processing"):
    started = asyncio.Event()
    release = asyncio.Event()
    tasks = set()
    counts = {}

    def response(ident, status):
        return {
            "id": ident,
            "audio_url": "https://example.org/audio.wav",
            "status": status,
            "text": "finished text" if status == "completed" else None,
            "error": "transcription failed" if status == "error" else None,
        }

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode().split("\r\n")
            method, path, _ = lines[0].split()
            headers = dict(
                line.lower().split(": ", 1) for line in lines[1:] if ": " in line
            )
            await reader.readexactly(int(headers.get("content-length", 0)))
            if method == "POST":
                body = response("job", "queued")
            else:
                ident = path.rsplit("/", 1)[-1]
                counts[ident] = counts.get(ident, 0) + 1
                if ident == "job" and counts[ident] == 1:
                    # Snapshot first, delay only its delivery. Server state
                    # advances to terminal before the second GET.
                    body = response(ident, old_status)
                    started.set()
                    await release.wait()
                else:
                    body = response(ident, terminal)
            data = json.dumps(body).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
                + data
            )
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
        yield f"http://127.0.0.1:{port}", started, release, counts
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


@pytest.mark.parametrize("terminal", ["completed", "error"])
@pytest.mark.parametrize("old_status", ["queued", "processing"])
async def test_delayed_poll_does_not_erase_returned_terminal_result(
    terminal, old_status
):
    async with poll_server(terminal, old_status) as (url, started, release, counts):
        # Real public submit -> returned AsyncTranscript, real default pool,
        # default 3s polling and 30s timeout. No SDK methods are patched.
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", base_url=url)
        ) as client:
            transcript = await aai.AsyncTranscriber(client=client).submit(
                "https://example.org/audio.wav"
            )
            old_waiter = asyncio.create_task(transcript.wait_for_completion())
            try:
                await asyncio.wait_for(started.wait(), 2)
                finished = await asyncio.wait_for(transcript.wait_for_completion(), 2)
                expected = (finished.status, finished.text, finished.error)
                assert finished is transcript
                assert finished.status.value == terminal
                release.set()
                await asyncio.wait({old_waiter}, timeout=0.2)
                observed = (finished.status, finished.text, finished.error)
                print(
                    f"terminal={terminal} stale={old_status} before={expected} after={observed} old_done={old_waiter.done()}"
                )
                assert observed == expected
                assert old_waiter.done()
                assert await old_waiter is finished
                assert counts["job"] == 2
                assert await transcript.wait_for_completion() is finished
                assert (finished.status, finished.text, finished.error) == expected
            finally:
                old_waiter.cancel()
                await asyncio.gather(old_waiter, return_exceptions=True)


async def test_cancelled_old_poll_does_not_affect_other_waiter():
    async with poll_server("completed") as (url, started, release, counts):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", base_url=url)
        ) as client:
            transcript = await aai.AsyncTranscriber(client=client).submit(
                "https://example.org/audio.wav"
            )
            old_waiter = asyncio.create_task(transcript.wait_for_completion())
            await asyncio.wait_for(started.wait(), 2)
            finished = await asyncio.wait_for(transcript.wait_for_completion(), 2)
            old_waiter.cancel()
            await asyncio.gather(old_waiter, return_exceptions=True)
            release.set()
            assert old_waiter.cancelled()
            assert finished.text == "finished text"
            assert finished.status == aai.TranscriptStatus.completed
            assert await transcript.wait_for_completion() is finished


async def test_unrelated_transcript_on_same_pool_is_not_blocked():
    async with poll_server("completed") as (url, started, release, counts):
        async with aai.AsyncClient(
            settings=aai.Settings(api_key="test", base_url=url)
        ) as client:
            transcriber = aai.AsyncTranscriber(client=client)
            transcript = await transcriber.submit("https://example.org/audio.wav")
            old_waiter = asyncio.create_task(transcript.wait_for_completion())
            try:
                await asyncio.wait_for(started.wait(), 2)
                unrelated = await asyncio.wait_for(transcriber.get_by_id("other"), 2)
                assert unrelated.id == "other"
                assert unrelated.text == "finished text"
                assert not old_waiter.done()
            finally:
                old_waiter.cancel()
                await asyncio.gather(old_waiter, return_exceptions=True)
                release.set()
