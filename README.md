<img src="https://github.com/AssemblyAI/assemblyai-python-sdk/blob/master/assemblyai.png?raw=true" width="500"/>

---

[![CI Passing](https://github.com/AssemblyAI/assemblyai-python-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/AssemblyAI/assemblyai-python-sdk/actions/workflows/test.yml)
[![GitHub License](https://img.shields.io/github/license/AssemblyAI/assemblyai-python-sdk)](https://github.com/AssemblyAI/assemblyai-python-sdk/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/assemblyai.svg)](https://badge.fury.io/py/assemblyai)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/assemblyai)](https://pypi.python.org/pypi/assemblyai/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/assemblyai)
[![AssemblyAI Twitter](https://img.shields.io/twitter/follow/AssemblyAI?label=%40AssemblyAI&style=social)](https://twitter.com/AssemblyAI)
[![AssemblyAI YouTube](https://img.shields.io/youtube/channel/subscribers/UCtatfZMf-8EkIwASXM4ts0A)](https://www.youtube.com/@AssemblyAI)
[![Discord](https://img.shields.io/discord/875120158014853141?logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F875120158014853141&style=social)
](https://assemblyai.com/discord)

# AssemblyAI's Python SDK

> _Build with AI models that can transcribe and understand audio_

With a single API call, get access to AI models built on the latest AI breakthroughs to transcribe and understand audio and speech data securely at large scale.

> **⚠️ WARNING**
> This SDK is intended for **testing and light usage only**. It is not
> recommended for use at scale or with production traffic. For best
> results, we recommend calling the AssemblyAI API directly via HTTP
> request. See our [official documentation](https://www.assemblyai.com/docs)
> for more information, including HTTP code examples.

## Claude Code

This repository includes a [`CLAUDE.md`](CLAUDE.md) file that provides context to Claude Code about this SDK — key APIs, common patterns, and gotchas. When you open this repo in Claude Code, it automatically reads this file to give better assistance.

# Overview

- [AssemblyAI's Python SDK](#assemblyais-python-sdk)
- [Overview](#overview)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Examples](#examples)
    - [**Core Examples**](#core-examples)
    - [**Speech Understanding Examples**](#speech-understanding-examples)
    - [**Streaming Examples**](#streaming-examples)
    - [**Change the default settings**](#change-the-default-settings)
  - [Playground](#playground)
- [Advanced](#advanced)
  - [How the SDK handles Default Configurations](#how-the-sdk-handles-default-configurations)
    - [Defining Defaults](#defining-defaults)
    - [Overriding Defaults](#overriding-defaults)
  - [Synchronous vs Asynchronous](#synchronous-vs-asynchronous)
  - [Getting the HTTP status code](#getting-the-http-status-code)
  - [Polling Intervals](#polling-intervals)
  - [Retrieving Existing Transcripts](#retrieving-existing-transcripts)
    - [Retrieving a Single Transcript](#retrieving-a-single-transcript)
    - [Retrieving Multiple Transcripts as a Group](#retrieving-multiple-transcripts-as-a-group)
    - [Retrieving Transcripts Asynchronously](#retrieving-transcripts-asynchronously)

# Documentation

Visit our [AssemblyAI API Documentation](https://www.assemblyai.com/docs) to get an overview of our models!

# Quick Start

## Installation

```bash
pip install -U assemblyai
```

## Examples

Before starting, you need to set the API key. If you don't have one yet, [**sign up for one**](https://www.assemblyai.com/dashboard/signup)!

```python
import assemblyai as aai

# set the API key
aai.settings.api_key = f"{ASSEMBLYAI_API_KEY}"
```

---

### **Core Examples**

<details>
  <summary>Transcribe a local audio file</summary>

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

audio_file = "./example.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    speaker_labels=True,
)

transcript = aai.Transcriber().transcribe(audio_file, config=config)

if transcript.status == aai.TranscriptStatus.error:
    raise RuntimeError(f"Transcription failed: {transcript.error}")
print(f"\nFull Transcript:\n\n{transcript.text}")
```

</details>

<details>
  <summary>Transcribe an URL</summary>

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    speaker_labels=True,
)

transcript = aai.Transcriber().transcribe(audio_file, config=config)

if transcript.status == aai.TranscriptStatus.error:
    raise RuntimeError(f"Transcription failed: {transcript.error}")
print(f"\nFull Transcript:\n\n{transcript.text}")
```

</details>

<details>
  <summary>Transcribe binary data</summary>

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

transcriber = aai.Transcriber()

# Binary data is supported directly:
transcript = transcriber.transcribe(data)

# Or: Upload data separately:
upload_url = transcriber.upload_file(data)
transcript = transcriber.transcribe(upload_url)
```

</details>

<details>
  <summary>Export subtitles of an audio file</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

srt = transcript.export_subtitles_srt(
  # Optional: Customize the maximum number of characters per caption
  chars_per_caption=32
  )

with open(f"transcript_{transcript.id}.srt", "w") as srt_file:
  srt_file.write(srt)

# vtt = transcript.export_subtitles_vtt()

# with open(f"transcript_{transcript_id}.vtt", "w") as vtt_file:
#   vtt_file.write(vtt)
```

</details>

<details>
  <summary>List all sentences and paragraphs</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

sentences = transcript.get_sentences()
for sentence in sentences:
  print(sentence.text)
  print()

paragraphs = transcript.get_paragraphs()
for paragraph in paragraphs:
  print(paragraph.text)
  print()
```

</details>

<details>
  <summary>Search for words in a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

# Set the words you want to search for
words = ["foo", "bar", "foo bar", "42"]

matches = transcript.word_search(words)

for match in matches:
  print(f"Found '{match.text}' {match.count} times in the transcript")
```

</details>

<details>
  <summary>Add custom spellings on a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True
)
config.set_custom_spelling(
  {
    "Gettleman": ["gettleman"],
    "SQL": ["Sequel"],
  }
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

print(transcript.text)
```

</details>

<details>
  <summary>Upload a file</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
upload_url = transcriber.upload_file(data)
```

</details>

<details>
  <summary>Delete a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

print(transcript.text)

transcript.delete_by_id(transcript.id)

transcript = aai.Transcript.get_by_id(transcript.id)
print(transcript.text)
```

</details>

<details>
  <summary>List transcripts</summary>

This returns a page of transcripts you created.

```python
import assemblyai as aai

transcriber = aai.Transcriber()

page = transcriber.list_transcripts()
print(page.page_details)  # Page details
print(page.transcripts)  # List of transcripts
```

You can apply filter parameters:

```python
params = aai.ListTranscriptParameters(
    limit=3,
    status=aai.TranscriptStatus.completed,
)
page = transcriber.list_transcripts(params)
```

You can also paginate over all pages by using the helper property `before_id_of_prev_url`.

The `prev_url` always points to a page with older transcripts. If you extract the `before_id`
of the `prev_url` query parameters, you can paginate over all pages from newest to oldest.

```python
transcriber = aai.Transcriber()

params = aai.ListTranscriptParameters()

page = transcriber.list_transcripts(params)
while page.page_details.before_id_of_prev_url is not None:
    params.before_id = page.page_details.before_id_of_prev_url
    page = transcriber.list_transcripts(params)
```

</details>

---

### **Speech Understanding Examples**

<details>
  <summary>PII Redact a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
).set_redact_pii(
    policies=[
        aai.PIIRedactionPolicy.person_name,
        aai.PIIRedactionPolicy.organization,
        aai.PIIRedactionPolicy.occupation,
    ],
    substitution=aai.PIISubstitutionPolicy.hash,
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

print(transcript.text)
```

To request a copy of the original audio file with the redacted information "beeped" out, set `redact_pii_audio=True` in the config.
Once the `Transcript` object is returned, you can access the URL of the redacted audio file with `get_redacted_audio_url`, or save the redacted audio directly to disk with `save_redacted_audio`.

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
).set_redact_pii(
    policies=[
        aai.PIIRedactionPolicy.person_name,
        aai.PIIRedactionPolicy.organization,
        aai.PIIRedactionPolicy.occupation,
    ],
    substitution=aai.PIISubstitutionPolicy.hash,
    redact_audio=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

print(transcript.text)
print(transcript.get_redacted_audio_url())
```

[Read more about PII redaction here.](https://www.assemblyai.com/docs/pii-redaction)

</details>
<details>
  <summary>Summarize the content of a transcript over time</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    auto_chapters=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

for chapter in transcript.chapters:
  print(f"{chapter.start}-{chapter.end}: {chapter.headline}")
```

[Read more about auto chapters here.](https://www.assemblyai.com/docs/speech-understanding/auto-chapters)

</details>

<details>
  <summary>Summarize the content of a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
  speech_models=["universal-3-pro", "universal-2"],
  language_detection=True,
  summarization=True,
  summary_model=aai.SummarizationModel.informative,
  summary_type=aai.SummarizationType.bullets
)

transcript = aai.Transcriber().transcribe(audio_file, config)

print(f"Transcript ID: ", transcript.id)
print(transcript.summary)
```

By default, the summarization model will be `informative` and the summarization type will be `bullets`. [Read more about summarization models and types here](https://www.assemblyai.com/docs/speech-understanding/summarization).

To change the model and/or type, pass additional parameters to the `TranscriptionConfig`:

```python
config=aai.TranscriptionConfig(
  summarization=True,
  summary_model=aai.SummarizationModel.catchy,
  summary_type=aai.SummarizationType.headline
)
```

</details>
<details>
  <summary>Detect sensitive content in a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    content_safety=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)

print(f"Transcript ID:", transcript.id)

for result in transcript.content_safety.results:
    print(result.text)
    print(f"Timestamp: {result.timestamp.start} - {result.timestamp.end}")

    # Get category, confidence, and severity.
    for label in result.labels:
        print(f"{label.label} - {label.confidence} - {label.severity}")  # content safety category

# Get the confidence of the most common labels in relation to the entire audio file.
for label, confidence in transcript.content_safety.summary.items():
    print(f"{confidence * 100}% confident that the audio contains {label}")

# Get the overall severity of the most common labels in relation to the entire audio file.
for label, severity_confidence in transcript.content_safety.severity_score_summary.items():
    print(f"{severity_confidence.low * 100}% confident that the audio contains low-severity {label}")
    print(f"{severity_confidence.medium * 100}% confident that the audio contains medium-severity {label}")
    print(f"{severity_confidence.high * 100}% confident that the audio contains high-severity {label}")
```

[Read more about the content safety categories.](https://www.assemblyai.com/docs/content-moderation)

By default, the content safety model will only include labels with a confidence greater than 0.5 (50%). To change this, pass `content_safety_confidence` (as an integer percentage between 25 and 100, inclusive) to the `TranscriptionConfig`:

```python
config=aai.TranscriptionConfig(
  content_safety=True,
  content_safety_confidence=80,  # only include labels with a confidence greater than 80%
)
```

</details>
<details>
  <summary>Analyze the sentiment of sentences in a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    sentiment_analysis=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

for sentiment_result in transcript.sentiment_analysis:
    print(sentiment_result.text)
    print(sentiment_result.sentiment)  # POSITIVE, NEUTRAL, or NEGATIVE
    print(sentiment_result.confidence)
    print(f"Timestamp: {sentiment_result.start} - {sentiment_result.end}")
```

If `speaker_labels` is also enabled, then each sentiment analysis result will also include a `speaker` field.

```python
# ...

config = aai.TranscriptionConfig(sentiment_analysis=True, speaker_labels=True)

# ...

for sentiment_result in transcript.sentiment_analysis:
  print(sentiment_result.speaker)
```

[Read more about sentiment analysis here.](https://www.assemblyai.com/docs/speech-understanding/sentiment-analysis)

</details>
<details>
  <summary>Identify entities in a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    entity_detection=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

for entity in transcript.entities:
    print(entity.text)
    print(entity.entity_type)
    print(f"Timestamp: {entity.start} - {entity.end}\n")
```

[Read more about entity detection here.](https://www.assemblyai.com/docs/speech-understanding/entity-detection)

</details>
<details>
  <summary>Detect topics in a transcript (IAB Classification)</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    iab_categories=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

# Get the parts of the transcript that were tagged with topics
for result in transcript.iab_categories.results:
    print(result.text)
    print(f"Timestamp: {result.timestamp.start} - {result.timestamp.end}")
    for label in result.labels:
        print(f"{label.label} ({label.relevance})")

# Get a summary of all topics in the transcript
for topic, relevance in transcript.iab_categories.summary.items():
    print(f"Audio is {relevance * 100}% relevant to {topic}")
```

[Read more about IAB classification here.](https://www.assemblyai.com/docs/speech-understanding/topic-detection)

</details>
<details>
  <summary>Identify important words and phrases in a transcript</summary>

```python
import assemblyai as aai

aai.settings.api_key = "<YOUR_API_KEY>"

# audio_file = "./local_file.mp3"
audio_file = "https://assembly.ai/wildfires.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    auto_highlights=True
)

transcript = aai.Transcriber().transcribe(audio_file, config)
print(f"Transcript ID:", transcript.id)

for result in transcript.auto_highlights.results:
    print(f"Highlight: {result.text}, Count: {result.count}, Rank: {result.rank}, Timestamps: {result.timestamps}")
```

[Read more about auto highlights here.](https://www.assemblyai.com/docs/speech-understanding/key-phrases)

</details>

---

### **Streaming Examples**

[Read more about our streaming service.](https://www.assemblyai.com/docs/streaming/universal-3-pro)

<details>
  <summary>Stream your microphone in real-time</summary>

```bash
pip install -U assemblyai
```

```python
import logging
from typing import Type

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
    TerminationEvent,
)

api_key = "<YOUR_API_KEY>"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    print(f"{event.transcript} ({event.end_of_turn})")

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(
        f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
    )

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error occurred: {error}")

def main():
    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=16000,
            speech_model="u3-rt-pro",
        )
    )

    try:
        client.stream(
            aai.extras.MicrophoneStream(sample_rate=16000)
        )
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
```

</details>

---

### **Change the default settings**

You'll find the `Settings` class with all default values in [types.py](./assemblyai/types.py).

<details>
  <summary>Change the default timeout and polling interval</summary>

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

# The HTTP timeout in seconds for general requests, default is 30.0
aai.settings.http_timeout = 60.0

# The polling interval in seconds for long-running requests, default is 3.0
aai.settings.polling_interval = 10.0
```

</details>

---

## Playground

Visit our Playground to try our all of our Speech AI models and LeMUR for free:

- [Playground](https://www.assemblyai.com/dashboard/playground/)

# Advanced

## How the SDK handles Default Configurations

### Defining Defaults

When no `TranscriptionConfig` is being passed to the `Transcriber` or its methods, it will use a default instance of a `TranscriptionConfig`.

If you would like to re-use the same `TranscriptionConfig` for all your transcriptions,
you can set it on the `Transcriber` directly:

```python
config = aai.TranscriptionConfig(punctuate=False, format_text=False)

transcriber = aai.Transcriber(config=config)

# will use the same config for all `.transcribe*(...)` operations
transcriber.transcribe("https://example.org/audio.wav")
```

### Overriding Defaults

You can override the default configuration later via the `.config` property of the `Transcriber`:

```python
transcriber = aai.Transcriber()

# override the `Transcriber`'s config with a new config
transcriber.config = aai.TranscriptionConfig(punctuate=False, format_text=False)
```

In case you want to override the `Transcriber`'s configuration for a specific operation with a different one, you can do so via the `config` parameter of a `.transcribe*(...)` method:

```python
config = aai.TranscriptionConfig(punctuate=False, format_text=False)
# set a default configuration
transcriber = aai.Transcriber(config=config)

transcriber.transcribe(
    "https://example.com/audio.mp3",
    # overrides the above configuration on the `Transcriber` with the following
    config=aai.TranscriptionConfig(speech_models=["universal-3-pro", "universal-2"], multichannel=True, disfluencies=True)
)
```

## Synchronous vs Asynchronous

Currently, the SDK provides two ways to transcribe audio files.

The synchronous approach halts the application's flow until the transcription has been completed.

The asynchronous approach allows the application to continue running while the transcription is being processed. The caller receives a [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html) object which can be used to check the status of the transcription at a later time.

You can identify those two approaches by the `_async` suffix in the `Transcriber`'s method name (e.g. `transcribe` vs `transcribe_async`).

## Getting the HTTP status code

There are two ways of accessing the HTTP status code:

- All custom AssemblyAI Error classes have a `status_code` attribute.
- The latest HTTP response is stored in `aai.Client.get_default().latest_response` after every API call. This approach works also if no Exception is thrown.

```python
transcriber = aai.Transcriber()

# Option 1: Catch the error
try:
    transcript = transcriber.submit("./example.mp3")
except aai.AssemblyAIError as e:
    print(e.status_code)

# Option 2: Access the latest response through the client
client = aai.Client.get_default()

try:
    transcript = transcriber.submit("./example.mp3")
except:
    print(client.last_response)
    print(client.last_response.status_code)
```

## Polling Intervals

By default we poll the `Transcript`'s status each `3s`. In case you would like to adjust that interval:

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

aai.settings.polling_interval = 1.0
```

## Retrieving Existing Transcripts

### Retrieving a Single Transcript

If you previously created a transcript, you can use its ID to retrieve it later.

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

transcript = aai.Transcript.get_by_id("<TRANSCRIPT_ID>")

print(transcript.id)
print(transcript.text)
```

### Retrieving Multiple Transcripts as a Group

You can also retrieve multiple existing transcripts and combine them into a single `TranscriptGroup` object. This allows you to perform operations on the transcript group as a single unit.

```python
import assemblyai as aai

aai.settings.base_url = "https://api.assemblyai.com"
aai.settings.api_key = "YOUR_API_KEY"

transcript_group = aai.TranscriptGroup.get_by_ids(["<TRANSCRIPT_ID_1>", "<TRANSCRIPT_ID_2>"])

```

### Retrieving Transcripts Asynchronously

Both `Transcript.get_by_id` and `TranscriptGroup.get_by_ids` have asynchronous counterparts, `Transcript.get_by_id_async` and `TranscriptGroup.get_by_ids_async`, respectively. These functions immediately return a `Future` object, rather than blocking until the transcript(s) are retrieved.

See the above section on [Synchronous vs Asynchronous](#synchronous-vs-asynchronous) for more information.
