# AssemblyAI Python SDK

Speech-to-text and audio intelligence SDK. Supports pre-recorded transcription, real-time streaming, and audio analysis features.

## Quick start

```bash
pip install -U assemblyai
```

```python
import os
import assemblyai as aai

aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

transcript = aai.Transcriber().transcribe(
    "https://example.com/audio.mp3",
    config=aai.TranscriptionConfig(
        speech_models=["universal-3-5-pro", "universal-2"],
        speaker_labels=True,
    ),
)

print(transcript.text)
for utterance in transcript.utterances:
    print(f"Speaker {utterance.speaker}: {utterance.text}")
```

## Auth

Set `ASSEMBLYAI_API_KEY` as an environment variable, or:

```python
aai.settings.api_key = "your-key"
```

## Key classes

- `aai.Transcriber` — Transcribe files, URLs, or streams. Methods: `transcribe()`, `transcribe_async()`, `submit()`, `list_transcripts()`
- `aai.TranscriptionConfig` — All transcription options: `speech_models`, `speaker_labels`, `sentiment_analysis`, `entity_detection`, `auto_chapters`, `content_safety`, `language_detection`, `summarization`, `word_boost`, `disfluencies`
- `aai.Transcript` — Result object with `.text`, `.status`, `.utterances`, `.words`, `.chapters`, `.entities`, `.sentiment_analysis`. Methods: `get_sentences()`, `get_paragraphs()`, `export_subtitles_srt()`, `export_subtitles_vtt()`
- `aai.SyncTranscriber` — Synchronous pre-recorded transcription: audio in, transcript out, one request (no polling). Methods: `transcribe()`, `transcribe_async()`
- `aai.SyncTranscriptionConfig` — Sync options: `model` (default `universal-3-5-pro`), `prompt`, `keyterms_prompt`, `conversation_context`, `language_codes`, `timestamps`, `sample_rate`, `channels`
- `aai.SyncTranscriptResponse` — Sync result: `.text`, `.words` (`SyncWord` with `confidence` always, `start`/`end` only when `timestamps=True`), `.confidence`, `.audio_duration_ms`, `.session_id`, `.request_time_ms`
- `assemblyai.streaming.v3.StreamingClient` — Real-time streaming with event-based API (threaded)
- `assemblyai.streaming.v3.AsyncStreamingClient` — Asyncio-native counterpart; same options/events

## Common patterns

**Transcribe a local file:**
```python
transcript = aai.Transcriber().transcribe("./recording.mp3")
```

**With multiple features:**
```python
config = aai.TranscriptionConfig(
    speech_models=["universal-3-5-pro", "universal-2"],
    speaker_labels=True,
    sentiment_analysis=True,
    entity_detection=True,
    auto_chapters=True,
    language_detection=True,
)
transcript = aai.Transcriber().transcribe(audio_url, config=config)
```

**PII redaction** (uses setter, not constructor):
```python
config = aai.TranscriptionConfig()
config.set_redact_pii(
    policies=[aai.PIIRedactionPolicy.email_address, aai.PIIRedactionPolicy.phone_number],
    substitution=aai.PIISubstitutionPolicy.hash,
)
```

**Retrieve existing transcript:**
```python
transcript = aai.Transcript.get_by_id("transcript-id")
```

## Sync transcription (pre-recorded, single request)

`SyncTranscriber` posts a whole audio file and returns the finished transcript in one
round trip — no job id, no polling, no status enum. It targets the sync API
(`sync.assemblyai.com`), distinct from `Transcriber`'s async job API. Use it for short
clips where you want the answer inline; use `Transcriber` for long-form audio, URLs, or
the rich audio-intelligence features (speaker labels, chapters, sentiment, …) the sync
API doesn't expose.

```python
import assemblyai as aai

aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

result = aai.SyncTranscriber().transcribe("./call.wav")
print(result.text, result.session_id)
for w in result.words:
    print(w.text, w.confidence)  # w.start/w.end need timestamps=True (see below)
```

**Input**: a local file path, raw `bytes`, or a binary file object. **Not** a URL —
pass a path/bytes or use `Transcriber` for URL ingestion.

**Config** (all optional):
```python
config = aai.SyncTranscriptionConfig(
    prompt="Transcribe verbatim. Preserve disfluencies.",  # max 4096 chars
    keyterms_prompt=["AssemblyAI", "Lemur", "U3-Pro"],     # max 2048 chars total
    language_codes=["es"],                                 # or e.g. ["en", "es"] for multilingual; defaults to English
)
result = aai.SyncTranscriber().transcribe("./call.wav", config=config)
```

**Conversation context**: `conversation_context` carries prior turns from the same
conversation so the model keeps continuity and proper-noun spelling across a multi-turn
exchange. List them oldest-first (most recent last); a single prior turn can be a bare
string. Capped at 100 turns / 4096 chars total — over-cap context is trimmed (oldest
turns first), not rejected; oldest turns are likewise dropped first when over the
model token budget.
```python
config = aai.SyncTranscriptionConfig(
    conversation_context=[
        "I'd like to book a flight to Denver.",
        "Sure, what date were you thinking?",
    ],
)
result = aai.SyncTranscriber().transcribe("./reply.wav", config=config)
```

**Language**: `language_codes` takes a list of ISO 639-1 codes — a single-element list
for monolingual audio or several codes for multilingual audio — and steers the default
prompt toward those languages; ignored when you pass a custom `prompt`.
Supported: en, es, de, fr, it, pt, tr, nl, sv, no, da, fi, hi, vi, ar, he, ja, ur, zh.

**Word timestamps** are opt-in. By default words carry `text` and `confidence` only —
`start`/`end` are `None`. `timestamps=True` computes accurate per-word timings for a
small latency cost:
```python
config = aai.SyncTranscriptionConfig(timestamps=True)
result = aai.SyncTranscriber().transcribe("./call.wav", config=config)
for w in result.words:
    print(w.text, w.start, w.end)  # milliseconds
```

**Raw PCM** (S16LE) needs `sample_rate` + `channels`; WAV reads them from its header.
Setting either field routes the audio as `audio/pcm`, and both must be present:
```python
config = aai.SyncTranscriptionConfig(sample_rate=16000, channels=1)
result = aai.SyncTranscriber().transcribe(raw_pcm_bytes, config=config)
```

**Concurrency**: `transcribe_async()` returns a `concurrent.futures.Future` (thread-based,
not asyncio) for fanning out a handful of files. (An asyncio-native `AsyncSyncTranscriber`
is a planned follow-up for high-concurrency servers and event-loop codebases.)

**Errors**: failures raise `aai.SyncTranscriptError` with `.status_code`, a
machine-readable `.error_code` — the snake_cased problem-details `title` from the
server (`bad_audio`, `audio_too_short`, `audio_too_large`, `capacity_exceeded`,
`inference_timeout`, …) — and `.retry_after` (seconds) on 429/503. Audio limits: 80 ms–120 s,
≤40 MB, 16-bit, mono/stereo, sample rate ∈ {8000, 16000, 22050, 24000, 32000, 44100, 48000}.

**Model**: `config.model` defaults to `"universal-3-5-pro"` and is sent as the `X-AAI-Model`
routing header (`aai.SyncSpeechModel.universal_3_5_pro`).

**Pre-warming the connection**: the sync API is one request/response, so a `transcribe()`
that connects on demand pays the full DNS + TCP + TLS handshake on the critical path —
a network round trip that, for a distant client, can rival the transcription. Call
`warm()` as soon as you know audio is coming (e.g. while it is still being recorded) to
spend that setup concurrently; the next `transcribe()` reuses the open connection.
```python
aai.settings.keepalive_expiry = 120          # optional: hold idle connections 120s

with aai.SyncTranscriber() as transcriber:
    transcriber.warm()                       # fire as recording starts
    audio = record_until_done()
    result = transcriber.transcribe(audio)   # reuses the hot connection
```
`warm()` returns `True` once the socket is open, `False` on a transport failure. The
warmed connection is reused while it stays in the pool — `settings.keepalive_expiry`
seconds, which defaults to httpx's 5s. For "warm at the start of a recording, send when
it ends," raise `keepalive_expiry` (e.g. to 120, the sync audio cap) so one `warm()`
covers the whole clip; otherwise call `warm()` shortly before `transcribe()` (it is
idempotent, so re-warming to refresh is fine). It only helps the *first* request: sync
calls share the client's connection pool, so later `transcribe()` calls already reuse the
open connection. Note `keepalive_expiry` applies to the whole client (sync and async).

## Streaming (real-time)

Use `universal-3-5-pro` as the streaming model — it's the flagship and what every example below targets. Two clients with identical option/event/handler surfaces: `StreamingClient` (threaded) and `AsyncStreamingClient` (asyncio).

**Handler contract — important**: every handler is called as `handler(client, event)`. Two positional args. Plain functions and `async def` functions both work for the async client; async handlers are awaited inline on the read task, so don't block — use `asyncio.create_task(...)` if you need to fan out work. Exceptions inside handlers are logged and swallowed.

**Event payloads** (what the second arg to each handler is):
- `Begin` → `BeginEvent(id: str, expires_at: datetime)`
- `Turn` → `TurnEvent(transcript: str, end_of_turn: bool, turn_is_formatted: bool, end_of_turn_confidence: float, words: list[Word], language_code: str | None, language_confidence: float | None, turn_order: int)`
- `Termination` → `TerminationEvent(audio_duration_seconds: int | None, session_duration_seconds: int | None)`
- `SpeechStarted` → `SpeechStartedEvent(timestamp: int)` (ms)
- `Warning` → `WarningEvent(warning_code: int, warning: str)`
- `Error` → `StreamingError` (an `Exception` subclass) with `.code: int | None`; `str(error)` is the message. Server-side errors come through `on_error` rather than being raised, and the payload is a `StreamingError`, **not** the wire `ErrorEvent` class.
- `LLMGatewayResponse` → `LLMGatewayResponseEvent(turn_order: int, transcript: str, data: Any)`
- `SpeakerRevision` → `SpeakerRevisionEvent(revisions: list[SpeakerRevisionItem])` — diarization-only. Sent once per offline-recluster resolve. Each `SpeakerRevisionItem(turn_order: int, speaker_label: str | None, words: list[Word])` is an earlier Turn whose labels changed (unchanged turns are omitted). For each item, match by `turn_order` against the original Turn and replace its per-word `speaker` (and the turn-level `speaker_label`) with the revision's values. Text and word timestamps are unchanged.

**Sync streaming:**
```python
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions, StreamingEvents,
    StreamingParameters, TurnEvent,
)

def on_turn(client, event: TurnEvent):
    print(event.transcript, "end_of_turn=", event.end_of_turn)

def on_error(client, error):
    print(f"Error {error.code}: {error}")

client = StreamingClient(StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"]))
client.on(StreamingEvents.Turn, on_turn)
client.on(StreamingEvents.Error, on_error)
client.connect(StreamingParameters(
    sample_rate=16000, speech_model="universal-3-5-pro",
))
try:
    client.stream(audio_bytes_or_iterable)
finally:
    client.disconnect(terminate=True)
```

**Async streaming** (preferred for voice agents and asyncio-native codebases):
```python
import asyncio
from assemblyai.streaming.v3 import (
    AsyncStreamingClient, StreamingClientOptions, StreamingEvents,
    StreamingParameters,
)

async def on_turn(client, event):
    print(event.transcript)

async def main():
    async with AsyncStreamingClient(
        StreamingClientOptions(api_key=os.environ["ASSEMBLYAI_API_KEY"])
    ) as client:
        client.on(StreamingEvents.Turn, on_turn)
        await client.connect(StreamingParameters(
            sample_rate=16000, speech_model="universal-3-5-pro",
        ))
        await client.stream(audio_async_generator)  # async iterable of bytes

asyncio.run(main())
```

**Microphone streaming** (both clients): `pip install "assemblyai[extras]"` for `pyaudio`, then pass `aai.extras.MicrophoneStream(sample_rate=16000)` to `client.stream(...)`.

**Voice-agent tuning** (knobs that matter most when building a voice agent):
```python
StreamingParameters(
    sample_rate=8000, speech_model="universal-3-5-pro", encoding="pcm_mulaw",  # telephony
    prompt="Transcribe verbatim with filler words and full punctuation.",
    agent_context="What is your account number?",  # agent's last reply; biases U3Pro context (U3Pro only)
    keyterms_prompt=["yes", "no", "uh-huh", "mm-hmm"],  # bias short acknowledgements
    min_turn_silence=200,         # ms; lower = snappier, may split phrases
    max_turn_silence=1000,        # ms; hard cap before forcing end-of-turn
    end_of_turn_confidence_threshold=0.7,
)
```
Use `pcm_s16le` (default) + `sample_rate=16000` for microphone capture.

`agent_context` (like `prompt`) can also be updated mid-stream after each agent
turn: `client.set_params(StreamingSessionParameters(agent_context="..."))`.

**Error codes**: `StreamingErrorCodes` is a `dict[int, str]` mapping wire codes to human messages. Use `.get(...)` for lookup — not enum-style attribute access:
```python
from assemblyai.streaming.v3 import StreamingErrorCodes

def on_error(client, error):
    message = StreamingErrorCodes.get(error.code, str(error))
    print(f"Streaming error {error.code}: {message}")
```
Common codes worth branching on: `4001` Not Authorized, `4002` Insufficient Funds, `4029` Client sent audio too fast, `4031` Session idle for too long.

**Events enum**: `StreamingEvents.{Begin, Turn, Termination, SpeechStarted, Error, Warning, LLMGatewayResponse, SpeakerRevision}`. Register a handler for each you care about; the call is the same shape: `client.on(StreamingEvents.Begin, on_begin)`, `client.on(StreamingEvents.Error, on_error)`, etc.

**Slow async handlers — fan-out pattern**: async handlers are awaited inline on the read task. If `on_turn` calls a slow LLM/TTS, ingestion stalls. Fan out and drain on shutdown:
```python
pending: set[asyncio.Task] = set()

async def on_turn(client, event):
    if not event.end_of_turn:
        return
    t = asyncio.create_task(slow_responder(event.transcript))
    pending.add(t); t.add_done_callback(pending.discard)

# After client.stream(...) returns, before leaving the `async with`:
if pending:
    await asyncio.gather(*pending, return_exceptions=True)
```

**Mint a temporary token without streaming** (typical FastAPI/server use):
```python
async with AsyncStreamingClient(StreamingClientOptions(api_key=MASTER_KEY)) as client:
    return await client.create_temporary_token(expires_in_seconds=60)
```
Use `async with` even when you never call `connect()` — `create_temporary_token` lazily opens an `httpx.AsyncClient` and `__aexit__` closes it. Without the context manager you leak the HTTP pool per request. The sync `StreamingClient.create_temporary_token` doesn't need this (no pool to close).

**Pass the token to the streaming client** via `StreamingClientOptions(token=...)` — same surface for `StreamingClient` and `AsyncStreamingClient`:
```python
async with AsyncStreamingClient(StreamingClientOptions(token=token_from_server)) as client:
    await client.connect(StreamingParameters(sample_rate=16000, speech_model="universal-3-5-pro"))
    await client.stream(audio)
```

**Other gotchas**:
- Don't pass `aai.extras.stream_file(...)` to `AsyncStreamingClient.stream()` — it uses blocking `time.sleep` and starves the read task. Use an `async def` generator with `await asyncio.sleep(...)` instead.
- `format_turns=True` enables punctuation/casing on confirmed end-of-turns. Toggle mid-session via `client.set_params(StreamingSessionParameters(format_turns=True))`.
- `AsyncStreamingClient` used as `async with` calls `disconnect(terminate=True)` on normal block exit and `disconnect(terminate=False)` on exception — no explicit `disconnect()` needed inside the block.

## Important gotchas

- **Always check status**: `if transcript.status == aai.TranscriptStatus.error` — accessing `.text` on a failed transcript returns None, not an exception
- **`speech_models` takes a list** with fallback ordering: `["universal-3-5-pro", "universal-2"]`
- **PII redaction uses `set_redact_pii()`**, not a constructor parameter
- **Streaming v3 lives in its own module**: `assemblyai.streaming.v3` (there is no other streaming API in this SDK). See the "Streaming (real-time)" section above.
- **Microphone streaming needs extras**: `pip install "assemblyai[extras]"` for `pyaudio`
- **`transcribe_async()` returns a `concurrent.futures.Future`**, not an asyncio coroutine
- **Timestamps are in milliseconds** throughout the SDK
- **Minimum Python**: 3.8+

## Dependencies

`httpx`, `pydantic`, `typing-extensions`, `websockets`. Optional: `pyaudio` via `[extras]`.

## Docs

- [Full documentation](https://www.assemblyai.com/docs)
- [API reference](https://www.assemblyai.com/docs/api-reference)
- [llms-full.txt](https://www.assemblyai.com/docs/llms-full.txt?lang=python) (Python-filtered docs for LLMs)
