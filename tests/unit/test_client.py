import assemblyai as aai


def test_reset_client_on_settings_change():
    """
    Test that the settings are reset when the global settings have changed.
    """
    aai.settings.api_key = "before"
    transcriber = aai.Transcriber()

    assert transcriber._client.settings.api_key == "before"

    # Reset it to "test" again. All other tests are also working with this value
    aai.settings.api_key = "test"
    transcriber = aai.Transcriber()

    assert transcriber._client.settings.api_key == "test"
