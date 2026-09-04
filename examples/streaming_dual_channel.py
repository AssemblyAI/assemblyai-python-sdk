"""Transcribe one desktop application and an optional microphone live."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from array import array
from collections.abc import Sequence

SAMPLE_RATE_HZ = 16_000


def pcm_s16le(samples: memoryview) -> bytes:
    """Convert normalized float samples to little-endian signed 16-bit PCM."""
    converted = array("h")
    converted.extend(
        -32_768 if sample <= -1.0 else 32_767 if sample >= 1.0 else int(sample * 32_767)
        for sample in samples
    )
    if sys.byteorder != "little":
        converted.byteswap()
    return converted.tobytes()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a running desktop application with AssemblyAI. "
            "Add --microphone to keep local speech on a separate channel."
        )
    )
    parser.add_argument(
        "--application",
        help="exact display name or application identifier to capture",
    )
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="also capture the default microphone",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="stop automatically after this many seconds",
    )
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than zero")
    return args


async def run(args: argparse.Namespace) -> None:
    import pocketstation as pks
    import pocketstation.aio as pks_aio
    from pocketstation.observations import SessionEventType

    from assemblyai.streaming.v3 import (
        AsyncChannelStreamer,
        AsyncRealTimeTranscriber,
        RealTimeEvents,
        RealTimeParameters,
        RealTimeTranscriberOptions,
        SpeechModel,
    )

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set ASSEMBLYAI_API_KEY before running this example")

    application_name = args.application
    if application_name is None:
        application_name = input("Application to transcribe: ").strip()
    if not application_name:
        raise ValueError("Application name must not be empty")

    session = pks_aio.Session(
        sample_rate_hz=SAMPLE_RATE_HZ,
        channels=1,
        frame_duration_ms=20,
    )
    application = session.capture(pks.Source.application(application_name))
    microphone = (
        session.capture(pks.Source.microphone_default()) if args.microphone else None
    )

    channels_by_stem = {int(application.id): "application"}
    if microphone is not None:
        channels_by_stem[int(microphone.id)] = "microphone"

    client = AsyncRealTimeTranscriber(RealTimeTranscriberOptions(api_key=api_key))
    mixer = (
        AsyncChannelStreamer(
            client,
            channels=["application", "microphone"],
            sample_rate=SAMPLE_RATE_HZ,
        )
        if microphone is not None
        else None
    )

    def on_turn(_client: object, event: object) -> None:
        transcript = getattr(event, "transcript", "")
        if not transcript:
            return
        channel = getattr(event, "channel", None) or "application"
        print(f"[{channel}] {transcript}")

    if mixer is None:
        client.on(RealTimeEvents.Turn, on_turn)
    else:
        mixer.on(RealTimeEvents.Turn, on_turn)

    async def start_transcription() -> None:
        await client.connect(
            RealTimeParameters(
                sample_rate=SAMPLE_RATE_HZ,
                speech_model=SpeechModel.universal_3_5_pro,
                speaker_labels=True,
            )
        )

    async def send_audio(frame: pks.AudioFrame) -> None:
        channel = channels_by_stem.get(int(frame.stem_id))
        if channel is None:
            raise RuntimeError(
                f"Received audio from undeclared stem {int(frame.stem_id)}"
            )
        samples = pcm_s16le(frame.samples)
        if mixer is None:
            await client.stream(samples)
        else:
            await mixer.stream(channel, samples)

    async def stop_transcription() -> None:
        if mixer is not None:
            await mixer.flush()
        await client.disconnect(terminate=True)

    transcription = pks_aio.Connector(
        start=start_transcription,
        send=send_audio,
        stop=stop_transcription,
    )
    application.send_to(transcription)
    if microphone is not None:
        microphone.send_to(transcription)

    running = await session.start()

    async def close_ended_channels() -> None:
        if mixer is None:
            return
        closed: set[str] = set()
        async for event in running.events:
            if event.kind != SessionEventType.SOURCE_FAILURE or event.stem_id is None:
                continue
            channel = channels_by_stem.get(int(event.stem_id))
            if channel is not None and channel not in closed:
                await mixer.close_channel(channel)
                closed.add(channel)

    event_task = asyncio.create_task(close_ended_channels())
    try:
        if args.duration is None:
            await asyncio.to_thread(input, "Press Enter to stop.\n")
        else:
            await asyncio.sleep(args.duration)
    finally:
        await running.stop()
        await event_task


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
