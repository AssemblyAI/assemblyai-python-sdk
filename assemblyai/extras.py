import threading
import time
from typing import BinaryIO, Generator, Optional
from warnings import warn

from . import api
from .client import Client


class AssemblyAIExtrasNotInstalledError(ImportError):
    def __init__(
        self,
        msg="""
        You must install the extras for this SDK to use this feature.
        Run `pip install "assemblyai[extras]"` to install the extras.
        Make sure to install `apt install portaudio19-dev` (Debian/Ubuntu) or
        `brew install portaudio` (MacOS) before installing the extras
        """,
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class MicrophoneStream:
    """
    An iterator of raw microphone audio chunks (~100ms each), suitable for
    passing to a streaming client's ``stream()`` method.

    :meth:`pause`, :meth:`resume`, and :meth:`close` are thread-safe: they may
    be called from another thread while a read is in flight.
    """

    def __init__(self, sample_rate: int = 44_100, device_index: Optional[int] = None):
        """
        Creates a stream of audio from the microphone.

        Args:
            sample_rate: The sample rate to record audio at.
            device_index: The index of the input device to use. If None, uses the default device.
        """
        try:
            import pyaudio
        except ImportError:
            raise AssemblyAIExtrasNotInstalledError

        self._pyaudio = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self._chunk_size = int(self.sample_rate * 0.1)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=device_index,
        )

        self._open = True
        self._paused = False
        self._closed = False
        # Serializes device reads against teardown: closing the PortAudio
        # stream while another thread is blocked in read() deadlocks, so
        # close() waits for any in-flight read to finish first.
        self._lock = threading.Lock()

    def __iter__(self):
        """
        Returns the iterator object.
        """

        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.

        While paused (see :meth:`pause`), the microphone is still drained so its
        input buffer doesn't overflow, but silence is yielded in place of the
        captured audio. This keeps the streaming session alive (avoiding an idle
        timeout) without forwarding what the mic picks up.
        """
        if not self._open:
            raise StopIteration

        with self._lock:
            # Re-check after acquiring: a concurrent close() may have torn the
            # stream down while this thread waited for the lock.
            if not self._open:
                raise StopIteration

            try:
                data = self._stream.read(self._chunk_size)
            except KeyboardInterrupt:
                raise StopIteration

        if self._paused:
            return b"\x00" * len(data)

        return data

    def pause(self) -> None:
        """
        Pause forwarding microphone audio.

        The stream stays open and keeps reading from the microphone so its
        internal buffer doesn't overflow, but :meth:`__next__` yields silence
        instead of the captured audio. Useful for muting the mic while a voice
        agent is speaking so its own output (e.g. TTS played back through the
        speakers) isn't transcribed and fed into a feedback loop.

        Thread-safe: may be called from any thread while another thread is
        reading from the stream. Takes effect within one chunk (~100ms).
        """
        self._paused = True

    def resume(self) -> None:
        """
        Resume forwarding live microphone audio after :meth:`pause`.

        Thread-safe, like :meth:`pause`.
        """
        self._paused = False

    @property
    def paused(self) -> bool:
        """Whether the stream is currently yielding silence (see :meth:`pause`)."""
        return self._paused

    def close(self):
        """
        Closes the stream.

        Thread-safe and idempotent: may be called from any thread, including
        while another thread is blocked in a read. Teardown waits for the
        in-flight chunk (~100ms) to finish first — closing the PortAudio
        stream mid-read deadlocks the reading thread. After close(), the
        iterator raises ``StopIteration``.
        """

        # Stop new reads before waiting on the in-flight one, so the reader
        # can't re-acquire the lock ahead of teardown.
        self._open = False

        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._stream.is_active():
                self._stream.stop_stream()

            self._stream.close()
            self._pyaudio.terminate()


def stream_file(
    filepath: str,
    sample_rate: int,
) -> Generator[bytes, None, None]:
    """
    Mimics a stream of audio data by reading it chunk by chunk from a file.

    NOTE: Only supports WAV/PCM16 files as of now.

    Args:
        filepath: The path to the file to stream.
        sample_rate: The sample rate of the audio file.

    Returns: A generator that yields chunks of audio data.
    """
    chunk_duration = 0.3
    with open(filepath, "rb") as f:
        while True:
            # send in 300ms segments (2 bytes per frame)
            data = f.read(int(sample_rate * chunk_duration) * 2)

            if not data:
                break

            yield data

            time.sleep(chunk_duration)


def file_from_stream(data: BinaryIO) -> str:
    """
    DeprecationWarning: `file_from_stream()` is deprecated and will be removed in 1.0.0. Use `Transcriber.upload_file()` instead.

    Uploads the given stream and returns the uploaded audio url.

    This function can be used to transcribe data that's already
    available in memory.

    Example:
    ```
    upload_url = aai.extras.file_from_stream(data)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(upload_url)
    ```

    Args:
        `data`: A file-like object (in binary mode)
    """
    warn(
        "`file_from_stream()` is deprecated and will be removed in 1.0.0. Use `Transcriber.upload_file()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return api.upload_file(
        client=Client.get_default().http_client,
        audio_file=data,
    )
