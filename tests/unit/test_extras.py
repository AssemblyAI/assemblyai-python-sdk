from unittest.mock import mock_open, patch

import pytest_mock

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


def test_microphone_stream_pause_resume(mocker: pytest_mock.MockerFixture):
    """
    A paused MicrophoneStream keeps draining the device but yields silence in
    place of captured audio, and resumes forwarding live audio afterwards.
    """
    import pyaudio

    live_chunk = b"\x01\x02\x03\x04" * 8
    fake_stream = mocker.MagicMock()
    fake_stream.read.return_value = live_chunk
    mocker.patch.object(pyaudio.PyAudio, "open", return_value=fake_stream)

    mic = aai.extras.MicrophoneStream(sample_rate=16000)

    # Live: captured audio is forwarded unchanged.
    assert mic.paused is False
    assert next(mic) == live_chunk

    # Paused: yields silence of the same length, but still reads from the device
    # (so the input buffer keeps draining and the session stays alive).
    mic.pause()
    assert mic.paused is True
    reads_before = fake_stream.read.call_count
    assert next(mic) == b"\x00" * len(live_chunk)
    assert fake_stream.read.call_count == reads_before + 1

    # Resume: live audio is forwarded again.
    mic.resume()
    assert mic.paused is False
    assert next(mic) == live_chunk


def test_microphone_stream_close_during_read_is_thread_safe(
    mocker: pytest_mock.MockerFixture,
):
    """
    close() may be called from another thread while a read is in flight (the
    natural companion to pause()/resume() in voice agents). Teardown must wait
    for the in-flight chunk instead of closing the PortAudio stream mid-read,
    which deadlocks the reading thread.
    """
    import threading
    import time

    import pyaudio

    chunk = b"\x00" * 16
    read_started = threading.Event()

    def slow_read(num_frames, *args, **kwargs):
        read_started.set()
        time.sleep(0.05)  # simulate the ~100ms blocking device read
        return chunk

    fake_stream = mocker.MagicMock()
    fake_stream.read.side_effect = slow_read
    fake_stream.is_active.return_value = True
    mocker.patch.object(pyaudio.PyAudio, "open", return_value=fake_stream)
    terminate = mocker.patch.object(pyaudio.PyAudio, "terminate")

    mic = aai.extras.MicrophoneStream(sample_rate=16000)

    def consume():
        for _ in mic:
            pass

    reader = threading.Thread(target=consume)
    reader.start()
    assert read_started.wait(timeout=2), "reader never reached the device read"

    started = time.monotonic()
    mic.close()  # concurrent with an in-flight read
    close_duration = time.monotonic() - started

    reader.join(timeout=2)
    assert not reader.is_alive(), "reader thread did not exit after close()"
    assert close_duration < 2, "close() blocked far longer than one chunk read"
    # close() was called mid-read (after read_started, during the 50ms sleep),
    # so a thread-safe close must have waited out the in-flight read rather
    # than tearing the stream down underneath it.
    assert close_duration >= 0.03, "close() did not wait for the in-flight read"

    # Teardown ran exactly once, and only after the in-flight read finished.
    fake_stream.stop_stream.assert_called_once()
    fake_stream.close.assert_called_once()
    terminate.assert_called_once()

    # Idempotent: a second close() is a no-op, and iteration stays ended.
    mic.close()
    fake_stream.close.assert_called_once()
    terminate.assert_called_once()
    assert next(mic, None) is None
