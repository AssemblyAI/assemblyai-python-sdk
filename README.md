<img src="https://github.com/AssemblyAI/assemblyai-python-sdk/blob/master/assemblyai.png?raw=true" width="500"/>

---
[![CI Passing](https://github.com/AssemblyAI/assemblyai-python-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/AssemblyAI/assemblyai-python-sdk/actions/workflows/test.yml)
[![GitHub License](https://img.shields.io/github/license/AssemblyAI/assemblyai-python-sdk)](https://github.com/AssemblyAI/assemblyai-python-sdk/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/assemblyai.svg)](https://badge.fury.io/py/assemblyai)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/assemblyai)](https://pypi.python.org/pypi/assemblyai/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/assemblyai)
[![AssemblyAI Twitter](https://img.shields.io/twitter/follow/AssemblyAI?label=%40AssemblyAI&style=social)](https://twitter.com/AssemblyAI)
[![AssemblyAI YouTube](https://img.shields.io/youtube/channel/subscribers/UCtatfZMf-8EkIwASXM4ts0A)](https://www.youtube.com/@AssemblyAI)

# AssemblyAI's Python SDK

> _Build with AI models that can transcribe and understand audio_

With a single API call, get access to AI models built on the latest AI breakthroughs to transcribe and understand audio and speech data securely at large scale.

# Overview

- [Documentation](#documentation)
- [Installation](#installation)
- [Example](#examples)
  - [Core Examples](#core-examples)
  - [LeMUR Examples](#lemur-examples)
  - [Audio Intelligence+ Examples](#audio-intelligence-examples)
- [Playgrounds](#playgrounds)
- [Advanced](#advanced-todo)


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
  <summary>Transcribe a local Audio File</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)
```
</details>

<details>
  <summary>Transcribe an URL</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3")

print(transcript.text)
```
</details>

<details>
  <summary>Export Subtitles of an Audio File</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3")

# in SRT format
print(transcript.export_subtitles_srt())

# in VTT format
print(transcript.export_subtitles_vtt())
```
</details>

<details>
  <summary>List all Sentences and Paragraphs</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3")

sentences = transcript.get_sentences()
for sentence in sentences:
  print(sentence.text)

paragraphs = transcript.get_paragraphs()
for paragraph in paragraphs:
  print(paragraph.text)
```
</details>

<details>
  <summary>Search for Words in a Transcript</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3")

matches = transcript.word_search(["price", "product"])

for match in matches:
  print(f"Found '{match.text}' {match.count} times in the transcript")
```
</details>

<details>
  <summary>Add Custom Spellings on a Transcript</summary>

```python
import assemblyai as aai

config = aai.TranscriptionConfig()
config.set_custom_spelling(
  {
    "Kubernetes": ["k8s"],
    "SQL": ["Sequel"],
  }
)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3", config)

print(transcript.text)
```
</details>

---
### **LeMUR Examples**

<details>
  <summary>Use LeMUR to Summarize Multiple Transcripts</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript_group = transcriber.transcribe_group(
    [
        "https://example.org/customer1.mp3",
        "https://example.org/customer2.mp3",
    ],
)

summary = transcript_group.lemur.summarize(context="Customers asking for cars", answer_format="TLDR")

print(summary)
```
</details>

<details>
  <summary>Use LeMUR to Get Feedback from the AI Coach on Multiple Transcripts</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript_group = transcriber.transcribe_group(
    [
        "https://example.org/interviewee1.mp3",
        "https://example.org/interviewee2.mp3",
    ],
)

feedback = transcript_group.lemur.ask_coach(context="Who was the best interviewee?")

print(feedback)
```
</details>

<details>
  <summary>Use LeMUR to Ask Questions on a Single Transcript</summary>

```python
import assemblyai as aai

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/customer.mp3")

# ask some questions
questions = [
    aai.LemurQuestion(question="What car was the customer interested in?"),
    aai.LemurQuestion(question="What price range is the customer looking for?"),
]

results = transcript.lemur.question(questions)

for result in result:
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
```
</details>

---

### **Audio Intelligence+ Examples**

<details>
  <summary>PII Redact a Transcript</summary>

```python
import assemblyai as aai

config = aai.TranscriptionConfig()
config.set_pii_redact(
  # What should be redacted
  policies=[
      aai.PIIRedactionPolicy.credit_card_number,
      aai.PIIRedactionPolicy.email_address,
      aai.PIIRedactionPolicy.location,
      aai.PIIRedactionPolicy.person_name,
      aai.PIIRedactionPolicy.phone_number,
  ],
  # How it should be redacted
  substitution=aai.PIISubstitutionPolicy.hash,
)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe("https://example.org/audio.mp3", config)
```
</details>


---

## Playgrounds

Visit one of our Playgrounds:

- [LeMUR Playground](https://www.assemblyai.com/playground/v2/source)
- [Transcription Playground](https://www.assemblyai.com/playground)


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
    config=aai.TranscriptionConfig(dual_channel=True, disfluencies=True)
)
```

## Synchronous vs Asynchronous

Currently, the SDK provides two ways to transcribe audio files.

The synchronous approach halts the application's flow until the transcription has been completed.

The asynchronous approach allows the application to continue running while the transcription is being processed. The caller receives a [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html) object which can be used to check the status of the transcription at a later time.

You can identify those two approaches by the `_async` suffix in the `Transcriber`'s method name (e.g. `transcribe` vs `transcribe_async`).
