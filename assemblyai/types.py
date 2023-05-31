from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, BaseSettings, Extra
from typing_extensions import Self


class AssemblyAIError(Exception):
    """
    Base exception for all AssemblyAI errors
    """


class TranscriptError(AssemblyAIError):
    """
    Error class when a transcription fails
    """


class LemurError(AssemblyAIError):
    """
    Error class when a Lemur request fails
    """


class Settings(BaseSettings):
    """
    Settings for the AssemblyAI client
    """

    api_key: Optional[str]
    "The API key to authenticate with"

    http_timeout: float = 15.0
    "The default HTTP timeout for general requests"

    base_url: str = "https://api.assemblyai.com/v2"
    "The base URL for the AssemblyAI API"

    class Config:
        env_prefix = "assemblyai_"


class TranscriptStatus(str, Enum):
    """
    Transcript status
    """

    queued = "queued"
    processing = "processing"
    completed = "completed"
    error = "error"


class LanguageCode(str, Enum):
    """
    Supported languages for Transcribing Audio
    """

    en = "en"
    "Global English"

    en_au = "en_au"
    "Australian English"

    en_uk = "en_uk"
    "British English"

    en_us = "en_us"
    "English (US)"

    de = "de"
    "German"

    fr = "fr"
    "French"

    hi = "hi"
    "Hindi"

    it = "it"
    "Italian"

    ja = "ja"
    "Japanese"

    es = "es"
    "Spanish"

    nl = "nl"
    "Dutch"

    pt = "pt"
    "Portuguese"


class WordBoost(str, Enum):
    low = "low"
    default = "default"
    high = "high"


class EntityType(str, Enum):
    """
    Used for AssemblyAI's Entity Detection feature.

    See: https://www.assemblyai.com/docs/audio-intelligence#entity-detection
    """

    medical_process = "medical_process"
    "Medical process, including treatments, procedures, and tests (e.g., heart surgery, CT scan)"

    medical_condition = "medical_condition"
    "Name of a medical condition, disease, syndrome, deficit, or disorder (e.g., chronic fatigue syndrome, arrhythmia, depression)"

    blood_type = "blood_type"
    "Blood type (e.g., O-, AB positive)"

    drug = "drug"
    "Medications, vitamins, or supplements (e.g., Advil, Acetaminophen, Panadol)"

    injury = "injury"
    "Bodily injury (e.g., I broke my arm, I have a sprained wrist)"

    number_sequence = "number_sequence"
    "A 'lazy' rule that will redact any sequence of numbers equal to or greater than 2"

    email_address = "email_address"
    "Email address (e.g., support@assemblyai.com)"

    date_of_birth = "date_of_birth"
    "Date of Birth (e.g., Date of Birth: March 7,1961)"

    phone_number = "phone_number"
    "Telephone or fax number"

    us_social_security_number = "us_social_security_number"
    "Social Security Number or equivalent"

    credit_card_number = "credit_card_number"
    "Credit card number"

    credit_card_expiration = "credit_card_expiration"
    "Expiration date of a credit card"

    credit_card_cvv = "credit_card_cvv"
    "Credit card verification code (e.g., CVV: 080)"

    date = "date"
    "Specific calendar date (e.g., December 18)"

    nationality = "nationality"
    "Terms indicating nationality, ethnicity, or race (e.g., American, Asian, Caucasian)"

    event = "event"
    "Name of an event or holiday (e.g., Olympics, Yom Kippur)"

    language = "language"
    "Name of a natural language (e.g., Spanish, French)"

    location = "location"
    "Any Location reference including mailing address, postal code, city, state, province, or country"

    money_amount = "money_amount"
    "Name and/or amount of currency (e.g., 15 pesos, $94.50)"

    person_name = "person_name"
    "Name of a person (e.g., Bob, Doug Jones)"

    person_age = "person_age"
    "Number associated with an age (e.g., 27, 75)"

    organization = "organization"
    "Name of an organization (e.g., CNN, McDonalds, University of Alaska)"

    political_affiliation = "political_affiliation"
    "Terms referring to a political party, movement, or ideology (e.g., Republican, Liberal)"

    occupation = "occupation"
    "Job title or profession (e.g., professor, actors, engineer, CPA)"

    religion = "religion"
    "Terms indicating religious affiliation (e.g., Hindu, Catholic)"

    drivers_license = "drivers_license"
    "Driver’s license number (e.g., DL# 356933-540)"

    banking_information = "banking_information"
    "Banking information, including account and routing numbers"


# EntityType and PIIRedactionPolicy share the same values
PIIRedactionPolicy = EntityType
"""
Used for AssemblyAI's PII Redaction feature.

See: https://www.assemblyai.com/docs/audio-intelligence#pii-redaction
"""


class PIISubstitutionPolicy(str, Enum):
    """
    Used for AssemblyAI's PII Redaction feature.

    See: https://www.assemblyai.com/docs/audio-intelligence#customize-how-redacted-pii-is-transcribed
    """

    hash = "hash"
    "PII that is detected is replaced with a hash - #. For example, I'm calling for John is replaced with ####. (Applied by default)"

    entity_name = "entity_name"
    "PII that is detected is replaced with the associated policy name. For example, John is replaced with [PERSON_NAME]. This is recommended for readability."


class SummarizationModel(str, Enum):
    """
    Used for AssemblyAI's Summarization feature.

    See: https://www.assemblyai.com/docs/audio-intelligence#summarization
    """

    informative = "informative"
    """
    Best for files with a single speaker such as presentations or lectures.

    Supported Summarization Types:
        - `bullets`
        - `bullets_verbose`
        - `headline`
        - `paragraph`

    Required Parameters:
        - `punctuate`: `True`
        - `format_text`: `True`
    """

    conversational = "conversational"
    """
    Best for any 2 person conversation such as customer/agent or interview/interviewee calls.

    Supported Summarization Types:
        - `bullets`
        - `bullets_verbose`
        - `headline`
        - `paragraph`

    Required Parameters:
        - `punctuate`: `True`
        - `format_text`: `True`
        - `speaker_labels` or `dual_channel` set to `True`
    """

    catchy = "catchy"
    """
    Best for creating video, podcast, or media titles.

    Supported Summarization Types:
        - `headline`
        - `gist`

    Required Parameters:
        - `punctuate`: `True`
        - `format_text`: `True`
    """


class SummarizationType(str, Enum):
    """
    Used for AssemblyAI's Summarization feature.

    See: https://www.assemblyai.com/docs/audio-intelligence#summarization
    """

    bullets = "bullets"
    "A bulleted summary with the most important points."

    bullets_verbose = "bullets_verbose"
    "A longer bullet point list summarizing the entire transcription text."

    gist = "gist"
    "A few words summarizing the entire transcription text."

    headline = "headline"
    "A single sentence summarizing the entire transcription text."

    paragraph = "paragraph"
    "A single paragraph summarizing the entire transcription text."


class RawTranscriptionConfig(BaseModel):
    language_code: LanguageCode = LanguageCode.en_us
    """
    The language of your audio file. Possible values are found in Supported Languages.

    The default value is `en_us`.
    """

    punctuate: Optional[bool]
    "Enable Automatic Punctuation"

    format_text: Optional[bool]
    "Enable Text Formatting"

    dual_channel: Optional[bool]
    "Enable Dual Channel transcription"

    webhook_url: Optional[str]
    "The URL we should send webhooks to when your transcript is complete."
    webhook_auth_header_name: Optional[str]
    "The name of the header that is sent when the `webhook_url` is being called."
    webhook_auth_header_value: Optional[str]
    "The value of the `webhook_auth_header_name` that is sent when the `webhook_url` is being called."

    audio_start_from: Optional[int]
    "The point in time, in milliseconds, to begin transcription from in your media file."
    audio_end_at: Optional[int]
    "The point in time, in milliseconds, to stop transcribing in your media file."

    word_boost: Optional[List[str]]
    "A list of custom vocabulary to boost accuracy for."
    boost_param: Optional[WordBoost]
    "The weight to apply to words/phrases in the word_boost array."

    filter_profanity: Optional[bool]
    "Filter profanity from the transcribed text."

    redact_pii: Optional[bool]
    "Redact PII from the transcribed text."
    redact_pii_audio: Optional[bool]
    "Generate a copy of the original media file with spoken PII 'beeped' out."
    redact_pii_policies: Optional[List[PIIRedactionPolicy]]
    "The list of PII Redaction policies to enable."
    redact_pii_sub: Optional[PIISubstitutionPolicy]
    "The replacement logic for detected PII."

    speaker_labels: Optional[bool]
    "Enable Speaker Diarization."

    # content_safety: bool = False
    # "Enable Content Safety Detection."

    # iab_categories: bool = False
    # "Enable Topic Detection."

    custom_spelling: Optional[List[Dict[str, List[str]]]]
    "Customize how words are spelled and formatted using to and from values"

    disfluencies: Optional[bool]
    "Transcribe Filler Words, like 'umm', in your media file."

    # sentiment_analysis: bool = False
    # "Enable Sentiment Analysis."

    # auto_chapters: bool = False
    # "Enable Auto Chapters."

    # entity_detection: bool = False
    # "Enable Entity Detection."

    summarization: Optional[bool]
    "Enable Summarization"
    summary_model: Optional[SummarizationModel]
    "The summarization model to use in case `summarization` is enabled"
    summary_type: Optional[SummarizationType]
    "The summarization type to use in case `summarization` is enabled"

    # auto_highlights: bool = False
    # "Detect important phrases and words in your transcription text."

    language_detection: Optional[bool]
    """
    Identify the dominant language that's spoken in an audio file, and route the file to the appropriate model for the detected language.

    Automatic Language Detection is supported for the following languages:

        - English
        - Spanish
        - French
        - German
        - Italian
        - Portuguese
        - Dutch
    """

    class Config:
        extra = Extra.allow


class TranscriptionConfig:
    def __init__(
        self,
        language_code: LanguageCode = LanguageCode.en_us,
        punctuate: Optional[bool] = None,
        format_text: Optional[bool] = None,
        dual_channel: Optional[bool] = None,
        webhook_url: Optional[str] = None,
        webhook_auth_header_name: Optional[str] = None,
        webhook_auth_header_value: Optional[str] = None,
        audio_start_from: Optional[int] = None,
        audio_end_at: Optional[int] = None,
        word_boost: List[str] = [],
        boost_param: Optional[WordBoost] = None,
        filter_profanity: Optional[bool] = None,
        redact_pii: Optional[bool] = None,
        redact_pii_audio: Optional[bool] = None,
        redact_pii_policies: Optional[PIIRedactionPolicy] = None,
        redact_pii_sub: Optional[PIISubstitutionPolicy] = None,
        speaker_labels: Optional[bool] = None,
        # content_safety: bool = False,
        # iab_categories: bool = False,
        custom_spelling: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
        disfluencies: Optional[bool] = None,
        # sentiment_analysis: bool = False,
        # auto_chapters: bool = False,
        # entity_detection: bool = False,
        summarization: Optional[bool] = None,
        summary_model: Optional[SummarizationModel] = None,
        summary_type: Optional[SummarizationType] = None,
        # auto_highlights: bool = False,
        language_detection: Optional[bool] = None,
        raw_transcription_config: Optional[RawTranscriptionConfig] = None,
    ) -> None:
        """
        Args:
            language_code: The language of your audio file. Possible values are found in Supported Languages.
            punctuate: Enable Automatic Punctuation
            format_text: Enable Text Formatting
            dual_channel: Enable Dual Channel transcription
            webhoook_url: The URL we should send webhooks to when your transcript is complete.
            webhook_auth_header_name: The name of the header that is sent when the `webhook_url` is being called.
            webhook_auth_header_value: The value of the `webhook_auth_header_name` that is sent when the `webhoook_url` is being called.
            audio_start_from: The point in time, in milliseconds, to begin transcription from in your media file.
            audio_end_at: The point in time, in milliseconds, to stop transcribing in your media file.
            word_boost: A list of custom vocabulary to boost accuracy for.
            boost_param: The weight to apply to words/phrases in the word_boost array.
            filter_profanity: Filter profanity from the transcribed text.
            redact_pii: Redact PII from the transcribed text.
            redact_pii_audio: Generate a copy of the original media file with spoken PII 'beeped' out.
            redact_pii_policies: The list of PII Redaction policies to enable.
            redact_pii_sub: The replacement logic for detected PII.
            speaker_labels: Enable Speaker Diarization.
            content_safety: Enable Content Safety Detection.
            iab_categories: Enable Topic Detection.
            custom_spelling: Customize how words are spelled and formatted using to and from values.
            disfluencies: Transcribe Filler Words, like 'umm', in your media file.
            sentiment_analysis: Enable Sentiment Analysis.
            auto_chapters: Enable Auto Chapters.
            entity_detection: Enable Entity Detection.
            summarization: Enable Summarization
            summary_model: The summarization model to use in case `summarization` is enabled
            summary_type: The summarization type to use in case `summarization` is enabled
            auto_highlights: Detect important phrases and words in your transcription text.
            language_detection: Identify the dominant language that’s spoken in an audio file, and route the file to the appropriate model for the detected language.
            raw_transcription_config: Create the config from a `RawTranscriptionConfig`
        """
        self._raw_transcription_config = raw_transcription_config

        if raw_transcription_config is None:
            self._raw_transcription_config = RawTranscriptionConfig()

        # explicit configurations have higher priority if `raw_transcription_config` has been passed as well
        self.language_code = language_code
        self.punctuate = punctuate
        self.format_text = format_text
        self.dual_channel = dual_channel
        self.set_webhook(
            webhook_url,
            webhook_auth_header_name,
            webhook_auth_header_value,
        )
        self.set_audio_slice(
            audio_start_from,
            audio_end_at,
        )
        self.set_word_boost(word_boost, boost_param)
        self.filter_profanity = filter_profanity
        self.set_redact_pii(
            redact_pii,
            redact_pii_audio,
            redact_pii_policies,
            redact_pii_sub,
        )
        self.speaker_labels = speaker_labels
        # self.content_safety = content_safety
        # self.iab_categories = iab_categories
        self.set_custom_spelling(custom_spelling, override=True)
        self.disfluencies = disfluencies
        # self.sentiment_analysis = sentiment_analysis
        # self.auto_chapters = auto_chapters
        # self.entity_detection = entity_detection
        self.set_summarize(
            summarization,
            summary_model,
            summary_type,
        )
        # self.auto_highlights = auto_highlights
        self.language_detection = language_detection

    @property
    def raw(self) -> RawTranscriptionConfig:
        return self._raw_transcription_config

    # region: Getters/Setters

    @property
    def language_code(self) -> LanguageCode:
        "The language code of the audio file."
        return self._raw_transcription_config.language_code

    @language_code.setter
    def language_code(self, language_code: LanguageCode) -> None:
        "Sets the language code of the audio file."

        self._raw_transcription_config.language_code = language_code

    @property
    def punctuate(self) -> Optional[bool]:
        "Returns the status of the Automatic Punctuation feature."

        return self._raw_transcription_config.punctuate

    @punctuate.setter
    def punctuate(self, enable: Optional[bool]) -> None:
        "Enable Automatic Punctuation feature."

        self._raw_transcription_config.punctuate = enable

    @property
    def format_text(self) -> Optional[bool]:
        "Returns the status of the Text Formatting feature."

        return self._raw_transcription_config.format_text

    @format_text.setter
    def format_text(self, enable: Optional[bool]) -> None:
        "Enables Formatting Text feature."

        self._raw_transcription_config.format_text = enable

    @property
    def dual_channel(self) -> Optional[bool]:
        "Returns the status of the Dual Channel transcription feature"

        return self._raw_transcription_config.dual_channel

    @dual_channel.setter
    def dual_channel(self, enable: Optional[bool]) -> None:
        "Enable Dual Channel transcription"

        self._raw_transcription_config.dual_channel = enable

    @property
    def webhook_url(self) -> Optional[str]:
        "The URL we should send webhooks to when your transcript is complete."

        return self._raw_transcription_config.webhook_url

    @property
    def webhook_auth_header_name(self) -> Optional[str]:
        "The name of the header that is sent when the `webhook_url` is being called."

        return self._raw_transcription_config.webhook_auth_header_name

    @property
    def webhook_auth_header_value(self) -> Optional[str]:
        "The value of the `webhook_auth_header_name` that is sent when the `webhook_url` is being called."

        return self._raw_transcription_config.webhook_auth_header_value

    @property
    def audio_start_from(self) -> Optional[int]:
        "Returns the point in time, in milliseconds, to begin transcription from in your media file."

        return self._raw_transcription_config.audio_start_from

    @property
    def audio_end_at(self) -> Optional[int]:
        "Returns the point in time, in milliseconds, to stop transcribing in your media file."

        return self._raw_transcription_config.audio_end_at

    @property
    def word_boost(self) -> Optional[List[str]]:
        "Returns the list of custom vocabulary to boost accuracy for."

        return self._raw_transcription_config.word_boost

    @property
    def boost_param(self) -> Optional[WordBoost]:
        "Returns how much weight is being applied when boosting custom vocabularies."

        return self._raw_transcription_config.boost_param

    @property
    def filter_profanity(self) -> Optional[bool]:
        "Returns the status of whether filtering profanity is enabled or not."

        return self._raw_transcription_config.filter_profanity

    @filter_profanity.setter
    def filter_profanity(self, enable: Optional[bool]) -> None:
        "Filter profanity from the transcribed text."

        self._raw_transcription_config.filter_profanity = enable

    @property
    def redact_pii(self) -> Optional[bool]:
        "Returns the status of the PII Redaction feature."

        return self._raw_transcription_config.redact_pii

    @property
    def redact_pii_audio(self) -> Optional[bool]:
        "Whether or not to generate a copy of the original media file with spoken PII 'beeped' out."

        return self._raw_transcription_config.redact_pii_audio

    @property
    def redact_pii_policies(self) -> Optional[List[PIIRedactionPolicy]]:
        "Returns a list of set of defined PII redaction policies."

        return self._raw_transcription_config.redact_pii_policies

    @property
    def redact_pii_sub(self) -> Optional[PIISubstitutionPolicy]:
        "Returns the replacement logic for detected PII."

        return self._raw_transcription_config.redact_pii_sub

    @property
    def speaker_labels(self) -> Optional[bool]:
        "Returns the status of the Speaker Diarization feature."

        return self._raw_transcription_config.speaker_labels

    @speaker_labels.setter
    def speaker_labels(self, enable: Optional[bool]) -> None:
        "Enable Speaker Diarization feature."

        self._raw_transcription_config.speaker_labels = enable

    # @property
    # def content_safety(self) -> bool:
    #     "Returns the status of the Content Safety feature."

    #     return self._raw_transcription_config.content_safety

    # @content_safety.setter
    # def content_safety(self, enable: bool) -> None:
    #     "Enable Content Safety feature."

    #     self._raw_transcription_config.content_safety = enable

    # @property
    # def iab_categories(self) -> bool:
    #     "Returns the status of the Topic Detection feature."

    #     return self._raw_transcription_config.iab_categories

    # @iab_categories.setter
    # def iab_categories(self, enable: bool) -> None:
    #     "Enable Topic Detection feature."

    #     self._raw_transcription_config.iab_categories = enable

    @property
    def custom_spelling(self) -> Optional[Dict[str, List[str]]]:
        "Returns the current set custom spellings."

        if self._raw_transcription_config.custom_spelling is None:
            return None

        custom_spellings = {}
        for custom_spelling in self._raw_transcription_config.custom_spelling:
            custom_spellings[custom_spelling["from"]] = custom_spelling["to"]

        return custom_spellings

    @property
    def disfluencies(self) -> Optional[bool]:
        "Returns whether to transcribing filler words is enabled or not."

        return self._raw_transcription_config.disfluencies

    @disfluencies.setter
    def disfluencies(self, enable: Optional[bool]) -> None:
        "Transcribe filler words, like 'umm', in your media file."

        self._raw_transcription_config.disfluencies = enable

        return self

    # @property
    # def sentiment_analysis(self) -> bool:
    #     "Returns the status of the Sentiment Analysis feature."

    #     return self._raw_transcription_config.sentiment_analysis

    # @sentiment_analysis.setter
    # def sentiment_analysis(self, enable: bool) -> None:
    #     "Enable Sentiment Analysis."

    #     self._raw_transcription_config.sentiment_analysis = enable

    # @property
    # def auto_chapters(self) -> bool:
    #     "Returns the status of the Auto Chapters feature."

    #     return self._raw_transcription_config.auto_chapters

    # @auto_chapters.setter
    # def auto_chapters(self, enable: bool) -> None:
    #     "Enable Auto Chapters."

    #     self._raw_transcription_config.auto_chapters = enable

    # @property
    # def entity_detection(self) -> bool:
    #     "Returns whether Entity Detection feature is enabled or not."

    #     return self._raw_transcription_config.entity_detection

    # @entity_detection.setter
    # def entity_detection(self, enable: bool) -> None:
    #     "Enable Entity Detection."

    #     self._raw_transcription_config.entity_detection = enable

    @property
    def summarization(self) -> Optional[bool]:
        "Returns whether the Summarization feature is enabled or not."

        return self._raw_transcription_config.summarization

    @property
    def summary_model(self) -> Optional[SummarizationModel]:
        "Returns the model of the Summarization feature."

        return self._raw_transcription_config.summary_model

    @property
    def summary_type(self) -> Optional[SummarizationType]:
        "Returns the type of the Summarization feature."

        return self._raw_transcription_config.summary_type

    # @property
    # def auto_highlights(self) -> bool:
    #     "Returns whether the Auto Highlights feature is enabled or not."

    #     return self._raw_transcription_config.auto_highlights

    # @auto_highlights.setter
    # def auto_highlights(self, enable: bool) -> None:
    #     "Detect important phrases and words in your transcription text."

    #     self._raw_transcription_config.auto_highlights = enable

    @property
    def language_detection(self) -> Optional[bool]:
        "Returns whether Automatic Language Detection is enabled or not."

        return self._raw_transcription_config.language_detection

    @language_detection.setter
    def language_detection(self, enable: Optional[bool]) -> None:
        """
        Identify the dominant language that's spoken in an audio file, and route the file to the appropriate model for the detected language.

        Automatic Language Detection is supported for the following languages:

            - English
            - Spanish
            - French
            - German
            - Italian
            - Portuguese
            - Dutch
        """

        self._raw_transcription_config.language_detection = enable

    # endregion

    # region: Convenience (helper) methods

    def set_casing_and_formatting(
        self,
        enable: bool = True,
    ) -> Self:
        """
        Whether to enable Automatic Punctuation and Text Formatting on the transcript.

        Args:
            enable: Enable Automatic Punctuation and Text Formatting
        """
        self._raw_transcription_config.punctuate = enable
        self._raw_transcription_config.format_text = enable

        return self

    def set_webhook(
        self,
        url: Optional[str],
        auth_header_name: Optional[str] = None,
        auth_header_value: Optional[str] = None,
    ) -> Self:
        """
        A webhook that is called on transcript completion.

        Args:
            url: The URL we should send webhooks to when your transcript is complete.
            auth_header_name: The name of the header that is sent when the `url` is being called.
            auth_header_value: The value of the `auth_header_name` that is sent when the `url` is being called.

        """

        if url is None:
            self._raw_transcription_config.webhook_url = None
            self._raw_transcription_config.webhook_auth_header_name = None
            self._raw_transcription_config.webhook_auth_header_value = None

            return self

        self._raw_transcription_config.url = url
        if auth_header_name and auth_header_value:
            self._raw_transcription_config.webhook_auth_header_name = auth_header_name
            self._raw_transcription_config.webhook_auth_header_value = auth_header_value

        return self

    def set_audio_slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Self:
        """
        Slice the audio to specify the start or end for transcription.

        Args:
            start: The point in time, in milliseconds, to begin transcription from in your media file.
            end: The point in time, in milliseconds, to stop transcribing in your media file.
        """

        self._raw_transcription_config.audio_start_from = start
        self._raw_transcription_config.audio_end_at = end

        return self

    def set_word_boost(
        self,
        words: List[str],
        boost: Optional[WordBoost] = WordBoost.default,
    ) -> Self:
        """
        Improve transcription accuracy when you know certain words or phrases will appear frequently in your audio file.

        Args:
            words: A list of words to improve accuracy on.
            boost: control how much weight should be applied to your keywords/phrases.

        Note: It's important to follow formatting guidelines for custom vocabulary to ensure the best results:
          - Remove all punctuation, except apostrophes, and make sure each word is in its spoken form.
          - Acronyms should have no spaces between letters.
          - Additionally, the model will still accept words with unique characters such as é,
            but will convert them to their ASCII equivalent.

        There are some limitations to the parameter. You can pass a maximum of 1,000 unique keywords/phrases in your list,
        and each of them must contain 6 words or less.
        """

        if not words:
            self._raw_transcription_config.word_boost = None
            self._raw_transcription_config.boost_param = None

            return self

        if not boost:
            self._raw_transcription_config.boost_param = WordBoost.default

        self._raw_transcription_config.word_boost = words
        self._raw_transcription_config.boost_param = boost

        return self

    def set_redact_pii(
        self,
        enable: bool = True,
        redact_audio: bool = False,
        policies: List[PIIRedactionPolicy] = [],
        substitution: PIISubstitutionPolicy = PIISubstitutionPolicy.hash,
    ) -> Self:
        """
        Enables Personal Identifiable Information (PII) Redaction feature.

        Args:
            enable: whether to enable or disable the PII Redaction feature.
            redact_audio: Generate a copy of the original media file with spoken PII 'beeped' out.
            policies: A list of PII redaction policies to enable.
            substitution: The replacement logic for detected PII (`PIISubstutionPolicy.hash` by default).
        """

        if not enable:
            self._raw_transcription_config.redact_pii = None
            self._raw_transcription_config.redact_pii_audio = None
            self._raw_transcription_config.redact_pii_policies = None
            self._raw_transcription_config.redact_pii_sub = None

            return self

        self._raw_transcription_config.redact_pii = True
        self._raw_transcription_config.redact_pii_audio = redact_audio
        self._raw_transcription_config.redact_pii_policies = policies
        self._raw_transcription_config.redact_pii_sub = substitution

        return self

    def set_custom_spelling(
        self,
        replacement: Optional[Dict[str, Union[str, Sequence[str]]]],
        override: bool = True,
    ) -> Self:
        """
        Customize how given words are being spelled or formatted in the transcription's text.

        Args:
            replacement: A dictionary that contains the replacement object (see below example)
            override: If `True` `replacement` gets overriden with the given `replacement` argument, otherwise merged.

        Example:
            ```
            config.custom_spelling({
                "AssemblyAI": "AssemblyAI",
                "Kubernetes": ["k8s", "kubernetes"]
            })
            ```
        """
        if replacement is None:
            self._raw_transcription_config.custom_spelling = None
            return self

        if self._raw_transcription_config.custom_spelling is None or override:
            self._raw_transcription_config.custom_spelling = []

        for to, from_ in replacement.items():
            if isinstance(from_, str):
                from_ = [from_]

            self._raw_transcription_config.custom_spelling.append(
                {
                    "from": list(from_),
                    "to": to,
                }
            )

        return self

    def set_summarize(
        self,
        enable: bool = True,
        model: Optional[SummarizationModel] = None,
        type: Optional[SummarizationType] = None,
    ) -> Self:
        """
        Enable Summarization.

        Args:
            enable: whether to enable to disable the Summarization feature.
            model: The summarization model to use
            type: The type of summarization to return
        """

        if not enable:
            self._raw_transcription_config.summarization = None
            self._raw_transcription_config.summary_model = None
            self._raw_transcription_config.summary_type = None

            return self

        self._raw_transcription_config.summarization = True
        self._raw_transcription_config.summary_model = model
        self._raw_transcription_config.summary_type = type

        return self

        # endregion


class ContentSafetyLabel(str, Enum):
    accidents = "accidents"
    "Any man-made incident that happens unexpectedly and results in damage, injury, or death."

    alcohol = "alcohol"
    "Content that discusses any alcoholic beverage or its consumption."

    financials = "financials"
    "Content that discusses any sensitive company financial information."

    crime_violence = "crime_violence"
    "Content that discusses any type of criminal activity or extreme violence that is criminal in nature."

    drugs = "drugs"
    "Content that discusses illegal drugs or their usage."

    gambling = "gambling"
    "Includes gambling on casino-based games such as poker, slots, etc. as well as sports betting."

    hate_speech = "hate_speech"
    """
    Content that is a direct attack against people or groups based on their
    sexual orientation, gender identity, race, religion, ethnicity, national origin, disability, etc.
    """

    health_issues = "health_issues"
    "Content that discusses any medical or health-related problems."

    manga = "manga"
    """
    Mangas are comics or graphic novels originating from Japan with some of the more popular series being
    "Pokemon", "Naruto", "Dragon Ball Z", "One Punch Man", and "Sailor Moon".
    """

    marijuana = "marijuana"
    "This category includes content that discusses marijuana or its usage."

    disasters = "disasters"
    """
    Phenomena that happens infrequently and results in damage, injury, or death.
    Such as hurricanes, tornadoes, earthquakes, volcano eruptions, and firestorms.
    """

    negative_news = "negative_news"
    """
    News content with a negative sentiment which typically will occur in the third person as an unbiased recapping of events.
    """

    nsfw = "nsfw"
    """
    Content considered "Not Safe for Work" and consists of content that a viewer would not want to be heard/seen in a public environment.
    """

    pornography = "pornography"
    "Content that discusses any sexual content or material."

    profanity = "profanity"
    "Any profanity or cursing."

    sensitive_social_issues = "sensitive_social_issues"
    """
    This category includes content that may be considered insensitive, irresponsible, or harmful
    to certain groups based on their beliefs, political affiliation, sexual orientation, or gender identity.
    """

    terrorism = "terrorism"
    """
    Includes terrorist acts as well as terrorist groups.
    Examples include bombings, mass shootings, and ISIS. Note that many texts corresponding to this topic may also be classified into the crime violence topic.
    """

    tobacco = "tobacco"
    "Text that discusses tobacco and tobacco usage, including e-cigarettes, nicotine, vaping, and general discussions about smoking."

    weapons = "weapons"
    "Text that discusses any type of weapon including guns, ammunition, shooting, knives, missiles, torpedoes, etc."


class Word(BaseModel):
    text: str
    start: int
    end: int
    confidence: float


class UtteranceWord(Word):
    channel: Optional[str]
    speaker: Optional[str]


class Utterance(UtteranceWord):
    words: List[UtteranceWord]


class Chapter(BaseModel):
    summary: str
    headline: str
    gist: str
    start: int
    end: int


class StatusResult(str, Enum):
    success = "success"
    unavailable = "unavailable"


class SentimentType(str, Enum):
    positive = "POSITIVE"
    neutral = "NEUTRAL"
    negative = "NEGATIVE"


class Timestamp(BaseModel):
    start: int
    end: int


class AutohighlightResult(BaseModel):
    count: int
    rank: float
    text: str
    timestamps: List[Timestamp]


class AutohighlightResponse(BaseModel):
    status: StatusResult
    results: Optional[List[AutohighlightResult]]


class ContentSafetyLabelResult(BaseModel):
    label: ContentSafetyLabel
    confidence: float
    severity: float


class ContentSafetySeverityScore(BaseModel):
    low: float
    medium: float
    high: float


class ContentSafetyResult(BaseModel):
    text: str
    labels: List[ContentSafetyLabelResult]
    timestamp: Timestamp


class ContentSafetyResponse(BaseModel):
    status: StatusResult
    results: Optional[List[ContentSafetyResult]]
    summary: Optional[Dict[str, float]]
    severity_score_summary: Optional[Dict[str, ContentSafetySeverityScore]]


class IABLabelResult(BaseModel):
    relevance: float
    label: str


class IABResult(BaseModel):
    text: str
    labels: List[IABLabelResult]
    timestamp: Timestamp


class IABResponse(BaseModel):
    status: StatusResult
    results: Optional[List[IABResult]]
    summary: Optional[Dict[str, float]]


class Sentiment(Word):
    sentiment: SentimentType


class Entity(BaseModel):
    entity_type: EntityType
    text: str
    start: int
    end: int


class WordSearchMatch(BaseModel):
    text: str
    "The word itself"

    count: int
    "The total amount of times the word is in the transcript"

    timestamps: List[Tuple[int, int]]
    "An array of timestamps structured as [start_time, end_time]"

    indexes: List[int]
    "An array of all index locations for that word within the words array of the completed transcript"


class WordSearchMatchResponse(BaseModel):
    total_count: int
    "Equals the total of all matched instances."

    matches: List[WordSearchMatch]
    "Contains a list/array of all matched words and associated data"


class Sentence(Word):
    words: List[Word]


class SentencesResponse(BaseModel):
    sentences: List[Sentence]
    confidence: float
    audio_duration: float


class Paragraph(Word):
    words: List[Word]


class ParagraphsResponse(BaseModel):
    paragraphs: List[Paragraph]
    confidence: float
    audio_duration: float


class BaseTranscript(BaseModel):
    """
    Available transcription features
    """

    language_code: LanguageCode = LanguageCode.en_us
    """
    The language of your audio file. Possible values are found in Supported Languages.

    The default value is `en_us`.
    """

    audio_url: str
    "The URL of your media file to transcribe."

    punctuate: Optional[bool]
    "Enable Automatic Punctuation"

    format_text: Optional[bool]
    "Enable Text Formatting"

    dual_channel: Optional[bool]
    "Enable Dual Channel transcription"

    webhook_url: Optional[str]
    "The URL we should send webhooks to when your transcript is complete."
    webhook_auth_header_name: Optional[str]
    "The name of the header that is sent when the `webhook_url` is being called."

    audio_start_from: Optional[int]
    "The point in time, in milliseconds, to begin transcription from in your media file."
    audio_end_at: Optional[int]
    "The point in time, in milliseconds, to stop transcribing in your media file."

    word_boost: Optional[List[str]]
    "A list of custom vocabulary to boost accuracy for."
    boost_param: Optional[WordBoost]
    "The weight to apply to words/phrases in the word_boost array."

    filter_profanity: Optional[bool]
    "Filter profanity from the transcribed text."

    redact_pii: Optional[bool]
    "Redact PII from the transcribed text."
    redact_pii_audio: Optional[bool]
    "Generate a copy of the original media file with spoken PII 'beeped' out."
    redact_pii_policies: Optional[List[PIIRedactionPolicy]]
    "The list of PII Redaction policies to enable."
    redact_pii_sub: Optional[PIISubstitutionPolicy]
    "The replacement logic for detected PII."

    speaker_labels: Optional[bool]
    "Enable Speaker Diarization."

    # content_safety: bool = False
    # "Enable Content Safety Detection."

    # iab_categories: bool = False
    # "Enable Topic Detection."

    custom_spelling: Optional[List[Dict[str, str]]]
    "Customize how words are spelled and formatted using to and from values"

    disfluencies: Optional[bool]
    "Transcribe Filler Words, like 'umm', in your media file."

    # sentiment_analysis: bool = False
    # "Enable Sentiment Analysis."

    # auto_chapters: bool = False
    # "Enable Auto Chapters."

    # entity_detection: bool = False
    # "Enable Entity Detection."

    summarization: Optional[bool]
    "Enable Summarization"
    summary_model: Optional[SummarizationModel]
    "The summarization model to use in case `summarization` is enabled"
    summary_type: Optional[SummarizationType]
    "The summarization type to use in case `summarization` is enabled"

    # auto_highlights: bool = False
    # "Detect important phrases and words in your transcription text."

    language_detection: Optional[bool]
    """
    Identify the dominant language that's spoken in an audio file, and route the file to the appropriate model for the detected language.

    Automatic Language Detection is supported for the following languages:

        - English
        - Spanish
        - French
        - German
        - Italian
        - Portuguese
        - Dutch
    """


class TranscriptRequest(BaseTranscript):
    """
    Transcript request schema
    """


class TranscriptResponse(BaseTranscript):
    """
    Transcript response schema
    """

    id: Optional[str]
    "The unique identifier of your transcription"

    status: TranscriptStatus
    "The status of your transcription. queued, processing, completed, or error"

    error: Optional[str]
    "The error message in case the transcription fails"

    text: Optional[str]
    "The text transcription of your media file"

    words: Optional[List[Word]]
    "A list of all the individual words transcribed"

    utterances: Optional[List[Utterance]]
    "When `dual_channel` or `speaker_labels` is enabled, a list of turn-by-turn utterances"

    confidence: Optional[float]
    "The confidence our model has in the transcribed text, between 0.0 and 1.0"

    audio_duration: Optional[float]
    "The duration of your media file, in seconds"

    webhook_status_code: Optional[int]
    "The status code we received from your server when delivering your webhook"
    webhook_auth: Optional[bool]
    "Whether the webhook was sent with an HTTP authentication header"

    # auto_highlights_result: Optional[AutohighlightResponse] = None
    # "The list of results when enabling Automatic Transcript Highlights"

    # content_safety_labels: Optional[ContentSafetyResponse] = None
    # "The list of results when Content Safety is enabled"

    # iab_categories_result: Optional[IABResponse] = None
    # "The list of results when Topic Detection is enabled"

    # chapters: Optional[List[Chapter]] = None
    # "When Auto Chapters is enabled, the list of Auto Chapters results"

    # sentiment_analysis_results: Optional[List[Sentiment]] = None
    # "When Sentiment Analysis is enabled, the list of Sentiment Analysis results"

    # entities: Optional[List[Entity]] = None
    # "When Entity Detection is enabled, the list of detected Entities"

    # def __init__(self, **data: Any):
    #     # cleanup the response before creating the object
    #     if data.get("iab_categories_result") == {}:
    #         data["iab_categories_result"] = None

    #     if data.get("content_safety_labels") == {}:
    #         data["content_safety_labels"] = None

    #     super().__init__(**data)


class LemurModel(str, Enum):
    lightning = "lightning"
    default = "default"


class LemurQuestionResult(BaseModel):
    question: str
    answer: str


class LemurQuestion(BaseModel):
    question: str
    context: Optional[Union[str, Dict[str, Any]]]
    answer_format: Optional[str]
    answer_options: Optional[List[str]]


class LemurQuestionRequest(BaseModel):
    transcript_ids: List[str]
    questions: List[LemurQuestion]
    model: LemurModel = LemurModel.default


class LemurQuestionResponse(BaseModel):
    response: List[LemurQuestionResult]
    model: LemurModel = LemurModel.default


class LemurSummaryRequest(BaseModel):
    transcript_ids: List[str]
    context: Optional[Union[str, Dict[str, Any]]]
    answer_format: Optional[str]
    model: LemurModel = LemurModel.default


class LemurSummaryResponse(BaseModel):
    response: str
    model: LemurModel = LemurModel.default


class LemurCoachRequest(BaseModel):
    transcript_ids: List[str]
    context: Union[str, Dict[str, Any]]
    model: LemurModel = LemurModel.default


class LemurCoachResponse(BaseModel):
    response: str
    model: LemurModel = LemurModel.default
