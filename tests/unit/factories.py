"""
Contains factories that are used for mocking certain requests/responses
from AssemblyAI's API.
"""

from enum import Enum
from functools import partial
from typing import Any, Callable, Dict

import factory
import factory.base

import assemblyai as aai
from assemblyai import types


class TimestampFactory(factory.Factory):
    class Meta:
        model = aai.Timestamp

    start = factory.Faker("pyint")
    end = factory.Faker("pyint")


class WordFactory(factory.Factory):
    class Meta:
        model = aai.Word

    text = factory.Faker("word")
    start = factory.Faker("pyint")
    end = factory.Faker("pyint")
    confidence = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)


class UtteranceWordFactory(WordFactory):
    class Meta:
        model = aai.UtteranceWord

    speaker = factory.Faker("name")


class UtteranceFactory(UtteranceWordFactory):
    class Meta:
        model = aai.Utterance

    words = factory.List([factory.SubFactory(UtteranceWordFactory)])


class ChapterFactory(factory.Factory):
    class Meta:
        model = types.Chapter

    summary = factory.Faker("sentence")
    headline = factory.Faker("sentence")
    gist = factory.Faker("sentence")
    start = factory.Faker("pyint")
    end = factory.Faker("pyint")


class BaseTranscriptFactory(factory.Factory):
    class Meta:
        model = types.BaseTranscript

    language_code = aai.LanguageCode.en
    audio_url = factory.Faker("url")
    punctuate = True
    format_text = True
    dual_channel = True
    webhook_url = None
    webhook_auth_header_name = None
    audio_start_from = None
    audio_end_at = None
    word_boost = None
    boost_param = None
    filter_profanity = False
    redact_pii = False
    redact_pii_audio = False
    redact_pii_policies = None
    redact_pii_sub = None
    speaker_labels = False
    content_safety = False
    iab_categories = False
    custom_spelling = None
    disfluencies = False
    sentiment_analysis = False
    auto_chapters = False
    entity_detection = False
    summarization = False
    summary_model = None
    summary_type = None
    auto_highlights = False
    language_detection = False


class TranscriptCompletedResponseFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptResponse

    id = factory.Faker("uuid4")
    status = aai.TranscriptStatus.completed
    error = None
    text = factory.Faker("text")
    words = factory.List([factory.SubFactory(WordFactory)])
    utterances = factory.List([factory.SubFactory(UtteranceFactory)])
    confidence = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)
    audio_duration = factory.Faker("pyint")
    webhook_auth = False
    webhook_status_code = None


class TranscriptQueuedResponseFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptResponse

    id = factory.Faker("uuid4")
    status = aai.TranscriptStatus.queued
    text = None
    words = None
    utterances = None
    confidence = None
    audio_duration = None


class TranscriptProcessingResponseFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptResponse

    id = factory.Faker("uuid4")
    status = aai.TranscriptStatus.processing
    text = None
    words = None
    utterances = None
    confidence = None
    audio_duration = None


class TranscriptErrorResponseFactory(TranscriptProcessingResponseFactory):
    class Meta:
        model = types.TranscriptResponse

    status = aai.TranscriptStatus.error
    error = "Aw, snap!"


class TranscriptRequestFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptRequest


class LemurQuestionResult(factory.Factory):
    class Meta:
        model = types.LemurQuestionResult

    question = factory.Faker("text")
    answer = factory.Faker("text")


class LemurQuestionResponse(factory.Factory):
    class Meta:
        model = types.LemurQuestionResponse

    response = factory.List(
        [
            factory.SubFactory(LemurQuestionResult),
            factory.SubFactory(LemurQuestionResult),
        ]
    )
    model = types.LemurModel.default


class LemurSummaryResponse(factory.Factory):
    class Meta:
        model = types.LemurSummaryResponse

    response = factory.Faker("text")
    model = types.LemurModel.default


class LemurAskCoachResponse(factory.Factory):
    class Meta:
        model = types.LemurSummaryResponse

    response = factory.Faker("text")
    model = types.LemurModel.default


class WordSearchMatchFactory(factory.Factory):
    class Meta:
        model = types.WordSearchMatch

    text = factory.Faker("text")
    count = factory.Faker("pyint")
    timestamps = [(123, 456)]
    indexes = [123, 456]


class WordSearchMatchResponseFactory(factory.Factory):
    class Meta:
        model = types.WordSearchMatchResponse

    total_count = factory.Faker("pyint")

    matches = factory.List([factory.SubFactory(WordSearchMatchFactory)])


class SentenceFactory(WordFactory):
    class Meta:
        model = types.Sentence

    words = factory.List([factory.SubFactory(WordFactory)])


class ParagraphFactory(SentenceFactory):
    class Meta:
        model = types.Paragraph


class SentencesResponseFactory(factory.Factory):
    class Meta:
        model = types.SentencesResponse

    sentences = factory.List([factory.SubFactory(SentenceFactory)])
    confidence = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)
    audio_duration = factory.Faker("pyint")


class ParagraphsResponseFactory(factory.Factory):
    class Meta:
        model = types.ParagraphsResponse

    paragraphs = factory.List([factory.SubFactory(ParagraphFactory)])
    confidence = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)
    audio_duration = factory.Faker("pyint")


def generate_dict_factory(f: factory.Factory) -> Callable[[], Dict[str, Any]]:
    """
    Creates a dict factory from the given *Factory class.

    Args:
        f: The factory to create a dict factory from.
    """

    def stub_is_list(stub: factory.base.StubObject) -> bool:
        try:
            return all(k.isdigit() for k in stub.__dict__.keys())
        except AttributeError:
            return False

    def convert_dict_from_stub(stub: factory.base.StubObject) -> Dict[str, Any]:
        stub_dict = stub.__dict__
        for key, value in stub_dict.items():
            if isinstance(value, factory.base.StubObject):
                stub_dict[key] = (
                    [convert_dict_from_stub(v) for v in value.__dict__.values()]
                    if stub_is_list(value)
                    else convert_dict_from_stub(value)
                )
            elif isinstance(value, Enum):
                stub_dict[key] = value.value
        return stub_dict

    def dict_factory(f, **kwargs):
        stub = f.stub(**kwargs)
        stub_dict = convert_dict_from_stub(stub)
        return stub_dict

    return partial(dict_factory, f)
