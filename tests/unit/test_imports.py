import os
import sys
from importlib import reload
from unittest.mock import mock_open, patch

import httpx
import pytest
import pytest_mock
from pytest_httpx import HTTPXMock

import assemblyai as aai
from assemblyai.api import ENDPOINT_UPLOAD


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


def test_import_sdk_and_use_extra_functions_without_extras_installed(
    httpx_mock: HTTPXMock,
):
    with ImportFailureMocker("pyaudio"):
        __reload_assesmblyai_module()

        local_file = os.urandom(10)
        expected_upload_url = "https://example.org/audio.wav"

        # patch the reading of a local file
        with patch("builtins.open", mock_open(read_data=local_file)):
            _ = aai.extras.stream_file(filepath="audio.wav", sample_rate=44_100)

        # mock the upload endpoint
        httpx_mock.add_response(
            url=f"{aai.settings.base_url}{ENDPOINT_UPLOAD}",
            status_code=httpx.codes.OK,
            method="POST",
            json={"upload_url": expected_upload_url},
            match_content=local_file,
        )

        upload_url = aai.extras.file_from_stream(local_file)
        assert upload_url == expected_upload_url


def test_import_sdk_and_use_MicrophoneStream_without_extras_installed():
    with ImportFailureMocker("pyaudio"):
        __reload_assesmblyai_module()

        with pytest.raises(aai.extras.AssemblyAIExtrasNotInstalledError):
            aai.extras.MicrophoneStream()


def test_import_sdk_and_use_MicrophoneStream_with_extras_installed(
    mocker: pytest_mock.MockerFixture,
):
    import pyaudio

    __reload_assesmblyai_module()

    mocker.patch.object(pyaudio.PyAudio, "open", return_value=None)
    aai.extras.MicrophoneStream()

    # Test succeeds if no failures
