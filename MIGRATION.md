# Migrating to 1.0

Two things in one guide: what genuinely breaks moving from 0.x to 1.0.0, and the patterns we recommend going forward for everything that still works the old way. Most 0.x code runs on 1.0 unchanged — read [Breaking changes](#breaking-changes) first (it is short), then adopt the rest at your own pace.

## TL;DR

| 0.x pattern | 1.0 best practice |
| --- | --- |
| `aai.Lemur(...)` | Removed. Feed `transcript.text` to the LLM of your choice |
| `from assemblyai.extras import MicrophoneStream` | Removed. Capture PCM yourself (`pyaudio`, `sounddevice`) and pass it to `stream(...)` |
| `pip install "assemblyai[extras]"` | `pip install -U assemblyai` |
| `StreamingClient`, `StreamingClientOptions` | `RealTimeTranscriber`, `RealTimeTranscriberOptions` |
| `from assemblyai.transcriber import Transcriber` | `from assemblyai.prerecorded.v2 import Transcriber` |
| `aai.Transcriber` / `aai.SyncTranscriber` (still fine) | `assemblyai.prerecorded.v2` / `assemblyai.sync.v1` imports |
| `aai.settings.api_key = "..."` as the only option | `Transcriber(api_key="...")` per client |
| `AsyncTranscriber()` left open | `async with AsyncTranscriber(...) as transcriber:` |
| `transcriber.transcribe(url)` polling forever | `transcriber.transcribe(url, poll_timeout=300)` |
| Assuming a returned transcript succeeded | Check `transcript.status` before reading `.text` |
| `SyncTranscriber().transcribe(...)` cold | `warm()` first, and raise `settings.keepalive_expiry` |
| `transcriber.transcribe(str(path))` | `transcriber.transcribe(path)` — `pathlib.Path` is accepted |

## Breaking changes

### LeMUR support removed

The LeMUR API and every `aai.Lemur*` name are gone from the SDK. There is no drop-in replacement in this package.

A transcript is still just text, so the migration is to send it to whichever LLM you already use:

```python
from assemblyai import TranscriptStatus
from assemblyai.prerecorded.v2 import Transcriber

transcript = Transcriber(api_key="YOUR_API_KEY").transcribe(
    "https://example.org/audio.wav"
)

if transcript.status == TranscriptStatus.error:
    raise RuntimeError(transcript.error)

prompt = f"Summarize this call transcript:\n\n{transcript.text}"
# ...hand `prompt` to your LLM client of choice.
```

### Audio-capture extras removed

`assemblyai.extras` (including `MicrophoneStream`) and the `[extras]` install option no longer exist. `pip install "assemblyai[extras]"` fails; use `pip install -U assemblyai`.

The SDK does not capture microphone audio. Bring your own capture — `pyaudio`, `sounddevice`, a loopback device, a file — and pass 16-bit PCM chunks to the streaming client:

```python
from assemblyai.streaming.v3 import RealTimeParameters, RealTimeTranscriber

transcriber = RealTimeTranscriber(api_key="YOUR_API_KEY")
transcriber.connect(RealTimeParameters(sample_rate=16_000))

# `chunks` is any iterable of 16-bit PCM frames from your capture library.
chunks = [b"\x00\x00" * 160]
transcriber.stream(chunks)
transcriber.disconnect()
```

## Still works, but modernize

### Streaming classes are now `RealTime*`

The streaming surface was renamed. Every former name remains bound to the same object, so `isinstance` checks and existing imports keep working — but new code should use the `RealTime*` names.

Before:

```python
from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions

client = StreamingClient(StreamingClientOptions(api_key="YOUR_API_KEY"))
```

After:

```python
from assemblyai.streaming.v3 import RealTimeTranscriber, RealTimeTranscriberOptions

transcriber = RealTimeTranscriber(RealTimeTranscriberOptions(api_key="YOUR_API_KEY"))
```

Full mapping — `StreamingClient`→`RealTimeTranscriber`, `AsyncStreamingClient`→`AsyncRealTimeTranscriber`, `StreamingClientOptions`→`RealTimeTranscriberOptions`, `StreamingParameters`→`RealTimeParameters`, `StreamingSessionParameters`→`RealTimeSessionParameters`, `StreamingEvents`→`RealTimeEvents`, `StreamingError`→`RealTimeError`, `StreamingErrorCodes`→`RealTimeErrorCodes`.

### Import from the versioned submodules

Each product lives in its own versioned subpackage — `assemblyai.prerecorded.v2`, `assemblyai.sync.v1`, `assemblyai.streaming.v3` — and importing a transcriber from its versioned path is the preferred style for new code: it says which product and which API version you are pinned to.

Before:

```python
import assemblyai as aai

transcriber = aai.Transcriber(api_key="YOUR_API_KEY")
sync_transcriber = aai.SyncTranscriber(api_key="YOUR_API_KEY")
```

After:

```python
from assemblyai.prerecorded.v2 import Transcriber
from assemblyai.sync.v1 import SyncTranscriber

transcriber = Transcriber(api_key="YOUR_API_KEY")
sync_transcriber = SyncTranscriber(api_key="YOUR_API_KEY")
```

The top-level `aai.Transcriber` / `aai.SyncTranscriber` names continue to work, as do the old flat module paths (`assemblyai.transcriber`, `assemblyai.sync`, `assemblyai.sync_api`) — nothing is deprecated.

Each subpackage exports only its own surface: `prerecorded.v2` gives you `Transcriber`, `AsyncTranscriber`, `Transcript`, `AsyncTranscript`, `TranscriptGroup`, and `TranscriptionConfig`; `sync.v1` gives you `SyncTranscriber`, `AsyncSyncTranscriber`, `SyncTranscriptionConfig`, `SyncTranscriptError`, and friends; `streaming.v3` gives you the `RealTime*` classes and every event type. Cross-cutting names — `TranscriptStatus`, `TranscriptError`, `Settings`, `Client`, `AsyncClient`, and the global `settings` — are **not** re-exported by the subpackages, so import those from the top-level package, mixing the two styles in one file as needed.

## New best practices in 1.0

### Pass `api_key=` where you build the client

Every transcriber and both streaming clients accept `api_key=`, so a key no longer has to travel through process-wide state.

Precedence, when combined with an explicit `client=` (or `options=` on the streaming clients): **`api_key=` wins**. The transcriber derives its own client from a copy of the given client's settings with the key replaced, and your client is left untouched.

```python
from assemblyai import Client, Settings
from assemblyai.prerecorded.v2 import Transcriber

transcriber = Transcriber(api_key="YOUR_API_KEY")

shared = Client(settings=Settings(api_key="TEAM_KEY", http_timeout=60.0))
reuses_shared = Transcriber(client=shared)                   # `shared` verbatim
per_tenant = Transcriber(client=shared, api_key="TENANT_KEY")
# `per_tenant` keeps http_timeout=60.0; `shared` still carries TEAM_KEY.
```

### Own the async client's lifecycle

The async transcribers hold an HTTP connection pool. Use them as async context managers so the pool is always released:

```python
import asyncio

from assemblyai.prerecorded.v2 import AsyncTranscriber


async def main():
    async with AsyncTranscriber(api_key="YOUR_API_KEY") as transcriber:
        transcript = await transcriber.transcribe("https://example.org/audio.wav")
        print(transcript.text)


asyncio.run(main())
```

`aclose()` is the explicit equivalent. A client you pass in with `client=` stays yours to close; anything the transcriber builds — including a client derived because you also passed `api_key=` — it closes itself.

### Bound your polling with `poll_timeout=`

`transcribe` polls until the transcript reaches a terminal status. Give it a deadline so a stuck job cannot hang a request handler:

```python
from assemblyai import TranscriptError
from assemblyai.prerecorded.v2 import Transcriber

transcriber = Transcriber(api_key="YOUR_API_KEY")

try:
    transcript = transcriber.transcribe(
        "https://example.org/audio.wav",
        poll_timeout=300,
    )
except TranscriptError as error:
    # The message carries the transcript id; the job keeps processing server-side.
    print(f"still running: {error}")
```

Available on `Transcriber.transcribe`, `Transcriber.transcribe_async`, and `AsyncTranscriber.transcribe`; omitting it keeps the unbounded behaviour. Resume later with `Transcript.get_by_id(transcript_id)` (`assemblyai.prerecorded.v2`).

### Check `transcript.status`

A transcription the server fails to complete is **returned, not raised**. `text` and `words` are `None` in that case, so check the status before reading the result:

```python
from assemblyai import TranscriptStatus
from assemblyai.prerecorded.v2 import Transcriber

transcriber = Transcriber(api_key="YOUR_API_KEY")
transcript = transcriber.transcribe("https://example.org/audio.wav")

if transcript.status == TranscriptStatus.error:
    raise RuntimeError(f"Transcription failed: {transcript.error}")

print(transcript.text)
```

### Warm the sync connection

`SyncTranscriber` is one request, so a cold DNS + TCP + TLS handshake lands on the critical path. Call `warm()` as soon as you know audio is coming, and raise `keepalive_expiry` so the connection survives until you send:

```python
import assemblyai as aai
from assemblyai.sync.v1 import SyncTranscriber

aai.settings.keepalive_expiry = 120  # seconds; matches the 120s sync audio cap

transcriber = SyncTranscriber(api_key="YOUR_API_KEY")
transcriber.warm()  # while the clip is still being recorded
# ...later: transcriber.transcribe("./call.wav")
```

`AsyncSyncTranscriber.warm()` is the awaitable equivalent.

### Pass `pathlib.Path` directly

Local files can be a `Path`, a `str` path, raw `bytes`/`bytearray`, or an open binary file — no `str()` wrapping needed:

```python
import pathlib

from assemblyai.prerecorded.v2 import Transcriber

transcriber = Transcriber(api_key="YOUR_API_KEY")
transcriber.transcribe(pathlib.Path("./call.wav"))
```

## What did *not* change

- `aai.settings.api_key = "..."` still works and is still the default for every client that is not given a key.
- Every former `Streaming*` name still imports and still resolves to the same object as its `RealTime*` counterpart.
- The flat import paths (`assemblyai.transcriber`, `assemblyai.sync`, `assemblyai.sync_api`) still work.
- No method signature was narrowed: every argument that worked in 0.x still works, and the new ones are optional keywords.
