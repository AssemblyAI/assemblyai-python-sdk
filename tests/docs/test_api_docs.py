from pathlib import Path

API_DOCS = Path(__file__).parents[2] / "docs" / "api"


def test_reference_contains_public_sdk_apis():
    html_files = list(API_DOCS.rglob("*.html"))
    assert html_files, "Build the API reference before running this test"

    reference = "\n".join(path.read_text(encoding="utf-8") for path in html_files)

    for public_name in (
        "Transcriber",
        "SyncTranscriber",
        "DictationTranscriber",
        "LLMGateway",
        "RealTimeTranscriber",
    ):
        assert public_name in reference

    for public_member in (
        "Transcriber.transcribe",
        "SyncTranscriber.transcribe",
        "DictationTranscriber.transcribe_live",
        "LLMGateway.chat",
        "RealTimeTranscriber.stream",
    ):
        assert f'id="{public_member}"' in reference

    assert "assemblyai._audio" not in reference
    assert "_audio.html" not in reference
