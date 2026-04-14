import assemblyai as aai


def test_speaker_identification_request_with_known_values_role():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.role,
        known_values=["Agent", "Customer"],
    )
    assert req.speaker_type == aai.SpeakerType.role
    assert req.known_values == ["Agent", "Customer"]
    assert req.speakers is None


def test_speaker_identification_request_with_known_values_name():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.name,
        known_values=["Alice", "Bob"],
    )
    assert req.speaker_type == aai.SpeakerType.name
    assert req.known_values == ["Alice", "Bob"]
    assert req.speakers is None


def test_speaker_identification_request_with_speakers_role():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.role,
        speakers=[
            {
                "role": "Operador",
                "description": "Human agent who starts the call with a standard greeting",
            },
            {
                "role": "IVR",
                "description": "Automated system playing recorded messages",
            },
            {
                "role": "Customer",
                "description": "The person who called the service center",
            },
        ],
    )
    assert req.speaker_type == aai.SpeakerType.role
    assert req.known_values is None
    assert len(req.speakers) == 3
    assert req.speakers[0]["role"] == "Operador"
    assert req.speakers[1]["role"] == "IVR"
    assert req.speakers[2]["role"] == "Customer"
    assert (
        req.speakers[0]["description"]
        == "Human agent who starts the call with a standard greeting"
    )


def test_speaker_identification_request_with_speakers_name():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.name,
        speakers=[
            {
                "name": "Michel Martin",
                "description": "Hosts the program and interviews the guests",
            },
            {
                "name": "Peter DeCarlo",
                "description": "Answers questions from the interview",
            },
        ],
    )
    assert req.speaker_type == aai.SpeakerType.name
    assert req.known_values is None
    assert len(req.speakers) == 2
    assert req.speakers[0]["name"] == "Michel Martin"
    assert req.speakers[1]["name"] == "Peter DeCarlo"


def test_speaker_identification_request_with_speakers_custom_properties():
    req = aai.SpeakerIdentificationRequest(
        speaker_type=aai.SpeakerType.name,
        speakers=[
            {
                "name": "Michel Martin",
                "description": "Hosts the program",
                "company": "NPR",
                "title": "Host Morning Edition",
            },
        ],
    )
    assert req.speakers[0]["company"] == "NPR"
    assert req.speakers[0]["title"] == "Host Morning Edition"


def test_speaker_identification_in_speech_understanding():
    config_args = {}
    config_args["speech_understanding"] = aai.SpeechUnderstandingRequest(
        request=aai.SpeechUnderstandingFeatureRequests(
            speaker_identification=aai.SpeakerIdentificationRequest(
                speaker_type=aai.SpeakerType.role,
                speakers=[
                    {
                        "role": "Operador",
                        "description": "Human agent who starts the call with a standard greeting",
                    },
                    {
                        "role": "IVR",
                        "description": "Automated system playing recorded messages",
                    },
                    {
                        "role": "Customer",
                        "description": "The person who called the service center",
                    },
                ],
            )
        )
    )
    si = config_args["speech_understanding"].request.speaker_identification
    assert si.speaker_type == aai.SpeakerType.role
    assert len(si.speakers) == 3
    assert si.speakers[0]["role"] == "Operador"
