import inspect

import assemblyai as aai


def test_configuration_drift():
    """
    Tests whether `TranscriptionConfig` drifts from `RawTranscriptionConfig` (properties, methods)
    """

    # a map of special setters that are defined in types.TranscriptionConfig
    special_setters = {
        "set_audio_slice",  # audio_start_from, audio_end_at
        "set_custom_spelling",  # custom_spelling
        "raw",  # access to the underlying raw config
        "set_word_boost",  # word boost setter
        "set_casing_and_formatting",  # punctuation, formatting setter
        "set_redact_pii",  # PII redaction
        "set_summarize",  # summarization
        "set_webhook",  # webhook
        "set_speaker_diarization",  # speaker diarization
        "set_content_safety",  # content safety
    }

    # get all members
    non_raw_members = inspect.getmembers(aai.TranscriptionConfig)

    # just retrieve the names
    raw_member_names = set(aai.RawTranscriptionConfig.__fields__.keys())
    non_raw_member_names = set(
        name for name, _ in non_raw_members if not name.startswith("_")
    )

    # get the differences
    diff_lhs = non_raw_member_names.difference(raw_member_names)
    diff_rhs = raw_member_names.difference(non_raw_member_names)
    differences = diff_lhs.union(diff_rhs)

    # check for the special setters
    differences = differences - special_setters

    # no differences: no drift.
    assert not differences
