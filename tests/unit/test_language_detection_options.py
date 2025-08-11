import assemblyai as aai


def test_language_detection_options_creation():
    """Test that LanguageDetectionOptions can be created with valid parameters."""
    options = aai.LanguageDetectionOptions(
        expected_languages=["en", "es", "fr"], fallback_language="en"
    )
    assert options.expected_languages == ["en", "es", "fr"]
    assert options.fallback_language == "en"


def test_language_detection_options_expected_languages_only():
    """Test that LanguageDetectionOptions can be created with only expected_languages."""
    options = aai.LanguageDetectionOptions(expected_languages=["en", "de"])
    assert options.expected_languages == ["en", "de"]
    assert options.fallback_language is None


def test_language_detection_options_fallback_language_only():
    """Test that LanguageDetectionOptions can be created with only fallback_language."""
    options = aai.LanguageDetectionOptions(fallback_language="es")
    assert options.expected_languages is None
    assert options.fallback_language == "es"


def test_language_detection_options_empty():
    """Test that LanguageDetectionOptions can be created with no parameters."""
    options = aai.LanguageDetectionOptions()
    assert options.expected_languages is None
    assert options.fallback_language is None


def test_transcription_config_with_language_detection_options():
    """Test that TranscriptionConfig accepts language_detection_options parameter."""
    options = aai.LanguageDetectionOptions(
        expected_languages=["en", "fr"], fallback_language="en"
    )

    config = aai.TranscriptionConfig(
        language_detection=True, language_detection_options=options
    )

    assert config.language_detection is True
    assert config.language_detection_options == options
    assert config.language_detection_options.expected_languages == ["en", "fr"]
    assert config.language_detection_options.fallback_language == "en"


def test_language_detection_options_property_getter():
    """Test the language_detection_options property getter."""
    options = aai.LanguageDetectionOptions(
        expected_languages=["ja", "ko"], fallback_language="ja"
    )

    config = aai.TranscriptionConfig()
    config.language_detection_options = options

    assert config.language_detection_options == options
    assert config.language_detection_options.expected_languages == ["ja", "ko"]
    assert config.language_detection_options.fallback_language == "ja"


def test_language_detection_options_property_setter():
    """Test the language_detection_options property setter."""
    config = aai.TranscriptionConfig()

    options = aai.LanguageDetectionOptions(
        expected_languages=["zh", "zh_cn"], fallback_language="zh"
    )
    config.language_detection_options = options

    assert config.language_detection_options == options


def test_language_detection_options_property_setter_none():
    """Test setting language_detection_options to None."""
    options = aai.LanguageDetectionOptions(fallback_language="en")
    config = aai.TranscriptionConfig(language_detection_options=options)

    # Verify it was set
    assert config.language_detection_options == options

    # Now set to None
    config.language_detection_options = None
    assert config.language_detection_options is None


def test_language_detection_options_in_raw_config():
    """Test that language_detection_options is properly set in the raw config."""
    options = aai.LanguageDetectionOptions(
        expected_languages=["en", "es"], fallback_language="en"
    )

    config = aai.TranscriptionConfig(language_detection_options=options)

    assert config.raw.language_detection_options == options


def test_set_language_detection():
    """Test the set_language_detection method."""
    config = aai.TranscriptionConfig().set_language_detection(
        confidence_threshold=0.8,
        expected_languages=["en", "fr"],
        fallback_language="en",
    )

    assert config.language_detection is True
    assert config.language_confidence_threshold == 0.8
    assert config.language_detection_options.expected_languages == ["en", "fr"]
    assert config.language_detection_options.fallback_language == "en"


def test_set_language_detection_disable():
    """Test disabling language detection clears all related options."""
    config = aai.TranscriptionConfig().set_language_detection(
        expected_languages=["en", "es"], fallback_language="en"
    )

    # Verify it was set
    assert config.language_detection is True
    assert config.language_detection_options is not None

    # Now disable
    config.set_language_detection(enable=False)

    assert config.language_detection is None
    assert config.language_confidence_threshold is None
    assert config.language_detection_options is None
