"""
Checks the built API reference in `docs/api/` for the public SDK surface.

Run it after `scripts/build_api_docs.sh`. It lives outside `tests/` because it
needs the built reference, which the regular test run does not build.
"""

import sys
from pathlib import Path

API_DOCS = Path(__file__).parent / "api"

# One page per documented module. The top-level `assemblyai` page re-exports
# most public names, so a module dropped from the build is only caught here.
MODULE_PAGES = (
    "assemblyai.html",
    "assemblyai/prerecorded/v2.html",
    "assemblyai/sync/v1.html",
    "assemblyai/dictation/v1.html",
    "assemblyai/llm_gateway/v1.html",
    "assemblyai/streaming/v3.html",
    "assemblyai/tts/v1.html",
)

PUBLIC_NAMES = (
    "Transcriber",
    "SyncTranscriber",
    "DictationTranscriber",
    "LLMGateway",
    "RealTimeTranscriber",
    "TTSSession",
)

PUBLIC_MEMBERS = (
    "Transcriber.transcribe",
    "SyncTranscriber.transcribe",
    "DictationTranscriber.transcribe_live",
    "LLMGateway.chat",
    "RealTimeTranscriber.stream",
    "TTSSession.speak",
)


def main() -> int:
    html_files = list(API_DOCS.rglob("*.html"))
    if not html_files:
        print("No API reference found: run scripts/build_api_docs.sh first")
        return 1

    reference = "\n".join(path.read_text(encoding="utf-8") for path in html_files)

    problems = [
        f"missing page {page}"
        for page in MODULE_PAGES
        if not (API_DOCS / page).is_file()
    ]
    problems += [f"missing {name}" for name in PUBLIC_NAMES if name not in reference]
    problems += [
        f"missing {member}"
        for member in PUBLIC_MEMBERS
        if f'id="{member}"' not in reference
    ]
    if "assemblyai._audio" in reference or "_audio.html" in reference:
        problems.append("the internal assemblyai._audio module is documented")

    for problem in problems:
        print(problem)
    if not problems:
        print(f"API reference OK ({len(html_files)} pages)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
