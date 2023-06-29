import sys
from importlib import reload

import pytest
import pytest_mock

import assemblyai as aai


class ImportFailureMocker:
    def __init__(self, module: str):
        self.module = module

    def find_spec(self, fullname, path, target=None):
        if fullname == self.module:
            raise ImportError

    def __enter__(self):
        # Remove module if already imported
        if self.module in sys.modules:
            del sys.modules[self.module]

        # Add self as first importer
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, type, value, traceback):
        # Remove self as importer
        sys.meta_path.pop(0)


def __reload_assesmblyai_module():
    reload(aai)
    aai.settings.api_key = "test"


def test_import_sdk_without_extras_installed():
    with ImportFailureMocker("pyaudio"):
        __reload_assesmblyai_module()
        # Test succeeds if no failures


def test_import_sdk_and_use_extras_without_extras_installed():
    with ImportFailureMocker("pyaudio"):
        __reload_assesmblyai_module()

        with pytest.raises(aai.extras.AssemblyAIExtrasNotInstalledError):
            aai.extras.MicrophoneStream()


def test_import_sdk_and_use_extras_with_extras_installed(
    mocker: pytest_mock.MockerFixture,
):
    import pyaudio

    __reload_assesmblyai_module()

    mocker.patch.object(pyaudio.PyAudio, "open", return_value=None)
    aai.extras.MicrophoneStream()

    # Test succeeds if no failures
