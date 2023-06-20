import time
from typing import Generator

try:
    import pyaudio
except ImportError:
    raise ImportError(
        "You must install the extras for this SDK to use this feature. "
        "Run `pip install assemblyai[extras]` to install the extras. "
        "Make sure to install `apt install portaudio19-dev` (Debian/Ubuntu) or "
        "`brew install portaudio` (MacOS) before installing the extras."
    )


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int = 44_100,
    ):
        """
        Creates a stream of audio from the microphone.

        Args:
            chunk_size: The size of each chunk of audio to read from the microphone.
            channels: The number of channels to record audio from.
            sample_rate: The sample rate to record audio at.
        """

        self._pyaudio = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self._chunk_size = int(self.sample_rate * 0.1)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
        )

        self._open = True

    def __iter__(self):
        """
        Returns the iterator object.
        """

        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        if not self._open:
            raise StopIteration

        try:
            return self._stream.read(self._chunk_size)
        except KeyboardInterrupt:
            raise StopIteration

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

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

    with open(filepath, "rb") as f:
        while True:
            data = f.read(int(sample_rate * 0.30) * 2)
            enough_data = ((len(data) / (16 / 8)) / sample_rate) * 1_000

            if not data or enough_data < 300.0:
                break

            yield data

            time.sleep(0.15)
