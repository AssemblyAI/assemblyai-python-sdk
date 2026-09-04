import importlib.util
from array import array
from pathlib import Path

import pytest

EXAMPLE_PATH = Path(__file__).parents[2] / "examples" / "streaming_dual_channel.py"


def load_example():
    spec = importlib.util.spec_from_file_location(
        "streaming_dual_channel_example", EXAMPLE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_capture_uses_application_only():
    example = load_example()

    args = example.parse_args([])

    assert args.application is None
    assert args.microphone is False
    assert args.duration is None


def test_capture_options_are_explicit():
    example = load_example()

    args = example.parse_args(
        ["--application", "Zoom", "--microphone", "--duration", "5"]
    )

    assert args.application == "Zoom"
    assert args.microphone is True
    assert args.duration == 5


def test_duration_must_be_positive():
    example = load_example()

    with pytest.raises(SystemExit):
        example.parse_args(["--duration", "0"])


def test_pcm_conversion_clips_and_uses_little_endian():
    example = load_example()
    samples = memoryview(array("f", [-2.0, -1.0, 0.0, 1.0, 2.0]))

    result = example.pcm_s16le(samples)

    assert result == b"\x00\x80\x00\x80\x00\x00\xff\x7f\xff\x7f"
