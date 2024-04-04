from unittest.mock import mock_open, patch

import assemblyai as aai


def test_stream_file_empty_file():
    """
    Test streaming of an empty file.
    """

    data = b""
    sample_rate = 44100

    m = mock_open(read_data=data)

    with patch("builtins.open", m), patch("time.sleep", return_value=None):
        chunks = list(aai.extras.stream_file("fake_path", sample_rate))

    # Expect no chunk
    assert len(chunks) == 0


def test_stream_file_small_file():
    """
    Tests streaming a file smaller than 300ms.
    """

    data = b"\x00" * int(0.2 * 44100) * 2
    sample_rate = 44100

    m = mock_open(read_data=data)

    with patch("builtins.open", m), patch("time.sleep", return_value=None):
        chunks = list(aai.extras.stream_file("fake_path", sample_rate))

    # Expecting one chunks because of no padding at the end
    expected_chunk_length = int(0.2 * sample_rate * 2)
    assert len(chunks) == 1
    assert len(chunks[0]) == expected_chunk_length
    assert chunks[0] == b"\x00" * expected_chunk_length


def test_stream_file_large_file():
    """
    Test streaming a file larger than 300ms.
    """

    data = b"\x00" * int(0.6 * 44100) * 2
    sample_rate = 44100

    m = mock_open(read_data=data)

    with patch("builtins.open", m), patch("time.sleep", return_value=None):
        chunks = list(aai.extras.stream_file("fake_path", sample_rate))

    # Expecting two chunks
    assert len(chunks) == 2


def test_stream_file_exact_file():
    """
    Test streaming a file exactly 300ms long.
    """

    data = b"\x00" * int(0.3 * 44100) * 2
    sample_rate = 44100

    m = mock_open(read_data=data)

    with patch("builtins.open", m), patch("time.sleep", return_value=None):
        chunks = list(aai.extras.stream_file("fake_path", sample_rate))

    # Expecting one chunk
    assert len(chunks) == 1
