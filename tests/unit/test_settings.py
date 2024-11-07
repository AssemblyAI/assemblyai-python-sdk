import os
from importlib import reload

import assemblyai as aai


def test_api_key_settings():
    """
    Tests that `ASSEMBLYAI_API_KEY` works correctly
    """
    tmp1 = os.environ.pop("ASSEMBLYAI_API_KEY", None)
    tmp2 = os.environ.pop("API_KEY", None)

    aai.settings.api_key = None
    reload(aai)
    assert aai.settings.api_key is None

    # this should not change the api key
    os.environ["API_KEY"] = "test"
    reload(aai)
    assert aai.settings.api_key is None

    # this should change the api key
    os.environ["ASSEMBLYAI_API_KEY"] = "test"
    reload(aai)
    assert aai.settings.api_key == "test"

    # reset
    if tmp1:
        os.environ["ASSEMBLYAI_API_KEY"] = tmp1
    else:
        os.environ.pop("ASSEMBLYAI_API_KEY", None)

    if tmp2:
        os.environ["API_KEY"] = tmp2
    else:
        os.environ.pop("API_KEY", None)

    reload(aai)
    aai.settings.api_key = "test"


def test_base_url_settings():
    """
    Tests that `ASSEMBLY_BASE_URL` works correctly
    """
    tmp1 = os.environ.pop("ASSEMBLYAI_BASE_URL", None)
    tmp2 = os.environ.pop("BASE_URL", None)

    aai.settings.base_url = "https://api.assemblyai.com"
    reload(aai)
    assert aai.settings.base_url == "https://api.assemblyai.com"

    # this should not change the base url
    os.environ["BASE_URL"] = "https://test.com"
    reload(aai)
    assert aai.settings.base_url == "https://api.assemblyai.com"

    # this should change the base url
    os.environ["ASSEMBLYAI_BASE_URL"] = "https://test.com"
    reload(aai)
    assert aai.settings.base_url == "https://test.com"

    # reset
    if tmp1:
        os.environ["ASSEMBLYAI_BASE_URL"] = tmp1
    else:
        os.environ.pop("ASSEMBLYAI_BASE_URL", None)
    if tmp2:
        os.environ["BASE_URL"] = tmp2
    else:
        os.environ.pop("BASE_URL", None)

    reload(aai)
    aai.settings.api_key = "test"
