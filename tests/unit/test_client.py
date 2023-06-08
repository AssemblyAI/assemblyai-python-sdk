import assemblyai as aai


def test_reset_client_on_settings_change():
    """
    Test that the settings are reset when the global settings have changed.
    """
    aai.settings.api_key = "before"
    transcriber = aai.Transcriber()

    assert transcriber._client.settings.api_key == "before"

    aai.settings.api_key = "after"
    transcriber = aai.Transcriber()

    assert transcriber._client.settings.api_key == "after"
