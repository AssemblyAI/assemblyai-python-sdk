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
    speaker = "1"
    channel = "1"


class UtteranceWordFactory(WordFactory):
    class Meta:
        model = aai.UtteranceWord

    speaker = "1"
    channel = "1"


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

    language_code = "en"
    audio_url = factory.Faker("url")
    punctuate = True
    format_text = True
    multichannel = None
    dual_channel = None
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
    speech_threshold = None


class BaseTranscriptResponseFactory(BaseTranscriptFactory):
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


class TranscriptDeletedResponseFactory(BaseTranscriptResponseFactory):
    language_code = None
    audio_url = "http://deleted_by_user"
    text = "Deleted by user."
    words = None
    utterances = None
    confidence = None
    punctuate = None
    format_text = None
    dual_channel = None
    multichannel = None
    webhook_url = "http://deleted_by_user"
    webhook_status_code = None
    webhook_auth = False
    # webhook_auth_header_name = None  # not yet supported in SDK
    speed_boost = None
    auto_highlights = None
    audio_start_from = None
    audio_end_at = None
    word_boost = None
    boost_param = None
    filter_profanity = None
    redact_pii_audio = None
    # redact_pii_quality = None # not yet supported in SDK
    redact_pii_policies = None
    redact_pii_sub = None
    speaker_labels = None
    error = None
    content_safety = None
    iab_categories = None
    content_safety_labels = None
    iab_categories = None
    language_detection = None
    custom_spelling = None
    # cluster_id = None  # not yet supported in SDK
    # custom_topics = None  # not yet supported in SDK
    # topics = None  # not yet supported in SDK
    speech_threshold = None
    chapters = None
    entities = None
    speakers_expected = None
    summary = None
    sentiment_analysis = None


class TranscriptCompletedResponseFactory(BaseTranscriptResponseFactory):
    pass


class TranscriptCompletedResponseFactoryBest(BaseTranscriptResponseFactory):
    speech_model = "best"


class TranscriptCompletedResponseFactoryNano(BaseTranscriptResponseFactory):
    speech_model = "nano"


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


class TranscriptErrorResponseFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptResponse

    status = aai.TranscriptStatus.error
    error = "Aw, snap!"


class TranscriptRequestFactory(BaseTranscriptFactory):
    class Meta:
        model = types.TranscriptRequest


class PageDetails(factory.Factory):
    class Meta:
        model = types.PageDetails

    current_url = factory.Faker("url")
    limit = 10
    next_url = None
    prev_url = None
    result_count = 2


class TranscriptItem(factory.Factory):
    class Meta:
        model = types.TranscriptItem

    audio_url = factory.Faker("url")
    created = factory.Faker("iso8601")
    id = factory.Faker("uuid4")
    resource_url = factory.Faker("url")
    status = aai.TranscriptStatus.completed
    completed = None
    error = None


class ListTranscriptResponse(factory.Factory):
    class Meta:
        model = types.ListTranscriptResponse

    page_details = factory.SubFactory(PageDetails)
    transcripts = factory.List(
        [
            factory.SubFactory(TranscriptItem),
            factory.SubFactory(TranscriptItem),
        ]
    )


class LemurUsage(factory.Factory):
    class Meta:
        model = types.LemurUsage

    input_tokens = factory.Faker("pyint")
    output_tokens = factory.Faker("pyint")


class LemurQuestionAnswer(factory.Factory):
    class Meta:
        model = types.LemurQuestionAnswer

    question = factory.Faker("text")
    answer = factory.Faker("text")


class LemurQuestionResponse(factory.Factory):
    class Meta:
        model = types.LemurQuestionResponse

    request_id = factory.Faker("uuid4")
    usage = factory.SubFactory(LemurUsage)
    response = factory.List(
        [
            factory.SubFactory(LemurQuestionAnswer),
            factory.SubFactory(LemurQuestionAnswer),
        ]
    )


class LemurSummaryResponse(factory.Factory):
    class Meta:
        model = types.LemurSummaryResponse

    request_id = factory.Faker("uuid4")
    usage = factory.SubFactory(LemurUsage)
    response = factory.Faker("text")


class LemurActionItemsResponse(factory.Factory):
    class Meta:
        model = types.LemurActionItemsResponse

    request_id = factory.Faker("uuid4")
    usage = factory.SubFactory(LemurUsage)
    response = factory.Faker("text")


class LemurTaskResponse(factory.Factory):
    class Meta:
        model = types.LemurTaskResponse

    request_id = factory.Faker("uuid4")
    usage = factory.SubFactory(LemurUsage)
    response = factory.Faker("text")


class LemurStringResponse(factory.Factory):
    class Meta:
        model = types.LemurStringResponse

    request_id = factory.Faker("uuid4")
    usage = factory.SubFactory(LemurUsage)
    response = factory.Faker("text")


class LemurPurgeResponse(factory.Factory):
    class Meta:
        model = types.LemurPurgeResponse

    request_id = factory.Faker("uuid4")
    request_id_to_purge = factory.Faker("uuid4")
    deleted = True


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
