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
        speech_models=["universal-3-pro", "universal-2"],
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
- `assemblyai.streaming.v3.StreamingClient` — Real-time streaming with event-based API

## Common patterns

**Transcribe a local file:**
```python
transcript = aai.Transcriber().transcribe("./recording.mp3")
```

**With multiple features:**
```python
config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
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

**Streaming v3:**
```python
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions,
    StreamingParameters, StreamingEvents,
)

client = StreamingClient(StreamingClientOptions(
    api_key=os.environ["ASSEMBLYAI_API_KEY"],
    api_host="streaming.assemblyai.com",
))
client.on(StreamingEvents.Turn, lambda turn: print(turn.text))
client.connect(StreamingParameters(
    sample_rate=16000,
    speech_model="u3-rt-pro",
))
```

**Retrieve existing transcript:**
```python
transcript = aai.Transcript.get_by_id("transcript-id")
```

## Important gotchas

- **Always check status**: `if transcript.status == aai.TranscriptStatus.error` — accessing `.text` on a failed transcript returns None, not an exception
- **`speech_models` takes a list** with fallback ordering: `["universal-3-pro", "universal-2"]`
- **PII redaction uses `set_redact_pii()`**, not a constructor parameter
- **Streaming v3 is a separate module**: `assemblyai.streaming.v3`, not the legacy `RealtimeTranscriber`
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
