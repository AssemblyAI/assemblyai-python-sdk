"""Unit tests for client-side dual/multi-channel streaming.

Covers the pure logic (energy VAD, VAD timeline, word/turn attribution,
unknown-channel resolution, the mono mixer) and the sync/async coordinators
against a fake client. No network I/O.
"""

import sys
from array import array

import pytest

from assemblyai.streaming.v3 import (
    AsyncChannelStreamer,
    ChannelAttributionOptions,
    ChannelStreamer,
    DualChannelTurnEvent,
    DualChannelWord,
    EnergyVad,
    StreamingEvents,
)
from assemblyai.streaming.v3.extras import (
    UNKNOWN_CHANNEL,
    VadFrame,
    VadTimeline,
    _ChannelMixer,
    attribute_turn,
    attribute_word,
    resolve_unknown_channels_by_speaker_history,
    resolve_unknown_channels_by_window,
    roll_up_turn_channel,
)

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _word(start: int, end: int, speaker=None, text="x") -> DualChannelWord:
    return DualChannelWord(
        start=start,
        end=end,
        confidence=1.0,
        text=text,
        word_is_final=True,
        speaker=speaker,
    )


def _turn(words) -> DualChannelTurnEvent:
    return DualChannelTurnEvent(
        type="Turn",
        turn_order=0,
        turn_is_formatted=False,
        end_of_turn=True,
        transcript=" ".join(w.text for w in words),
        end_of_turn_confidence=0.9,
        words=list(words),
    )


def _pcm(value: int, n: int) -> bytes:
    """``n`` samples of constant int16 ``value`` as little-endian PCM bytes."""
    samples = array("h", [value] * n)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


def _active_frame(channel: str, ts: float, rms: float = 0.5) -> VadFrame:
    return VadFrame(ts=ts, channel=channel, active=True, rms=rms)


# --------------------------------------------------------------------------- #
# EnergyVad
# --------------------------------------------------------------------------- #
def test_energy_vad_silence_is_inactive():
    vad = EnergyVad()
    result = vad.process([0.0] * 320)
    assert result.active is False
    assert result.energy == 0.0


def test_energy_vad_detects_speech_above_noise():
    vad = EnergyVad()
    # First a quiet frame to seat the (low) noise floor, then a loud one.
    vad.process([0.0] * 320)
    result = vad.process([0.3] * 320)
    assert result.active is True
    assert result.energy == pytest.approx(0.3, rel=1e-6)


def test_energy_vad_hangover_keeps_active_after_speech():
    vad = EnergyVad(hangover_frames=3)
    vad.process([0.3] * 320)  # speech -> sets hangover
    # Three quiet frames stay active by hangover, the fourth goes inactive.
    assert vad.process([0.0] * 320).active is True
    assert vad.process([0.0] * 320).active is True
    assert vad.process([0.0] * 320).active is True
    assert vad.process([0.0] * 320).active is False


def test_energy_vad_reset():
    vad = EnergyVad(hangover_frames=5)
    vad.process([0.5] * 320)
    vad.reset()
    # After reset, a silent frame should be inactive (no lingering hangover).
    assert vad.process([0.0] * 320).active is False


# --------------------------------------------------------------------------- #
# VadTimeline
# --------------------------------------------------------------------------- #
def test_timeline_returns_frames_in_window():
    tl = VadTimeline(window_ms=30_000)
    for ts in (0, 100, 200, 300, 400):
        tl.push_frame(_active_frame("mic", ts))
    frames = tl.frames_in_window(100, 300)
    assert [f.ts for f in frames] == [100, 200, 300]


def test_timeline_prunes_beyond_window():
    tl = VadTimeline(window_ms=200)
    for ts in (0, 100, 200, 1000):
        tl.push_frame(_active_frame("mic", ts))
    # Frames older than ts(1000) - 200 = 800 are pruned from the active head.
    assert all(f.ts >= 800 for f in tl.frames_in_window(0, 2000))


# --------------------------------------------------------------------------- #
# word / turn attribution
# --------------------------------------------------------------------------- #
def test_attribute_word_unknown_when_no_energy():
    tl = VadTimeline(30_000)
    assert attribute_word(_word(0, 100), tl, dominance_ratio=4.0) == UNKNOWN_CHANNEL


def test_attribute_word_single_channel_wins():
    tl = VadTimeline(30_000)
    for ts in (10, 30, 50):
        tl.push_frame(_active_frame("mic", ts))
    assert attribute_word(_word(0, 100), tl, dominance_ratio=4.0) == "mic"


def test_attribute_word_dominant_channel_by_ratio():
    tl = VadTimeline(30_000)
    tl.push_frame(_active_frame("mic", 10, rms=1.0))
    tl.push_frame(_active_frame("mic", 20, rms=1.0))
    tl.push_frame(_active_frame("system", 30, rms=0.1))
    # mic total 2.0 vs system 0.1 -> beats 4x ratio.
    assert attribute_word(_word(0, 100), tl, dominance_ratio=4.0) == "mic"


def test_attribute_word_close_scores_are_unknown():
    tl = VadTimeline(30_000)
    tl.push_frame(_active_frame("mic", 10, rms=1.0))
    tl.push_frame(_active_frame("system", 20, rms=0.9))
    # Too close to clear the 4x ratio -> unknown (resolution may back-fill).
    assert attribute_word(_word(0, 100), tl, dominance_ratio=4.0) == UNKNOWN_CHANNEL


def test_attribute_word_dominance_ratio_is_a_real_knob():
    tl = VadTimeline(30_000)
    tl.push_frame(_active_frame("mic", 10, rms=1.0))
    tl.push_frame(_active_frame("system", 20, rms=0.9))
    # Same frames, looser ratio -> mic now clears the bar and wins.
    assert attribute_word(_word(0, 100), tl, dominance_ratio=1.1) == "mic"


def test_attribute_word_exact_tie_is_unknown():
    tl = VadTimeline(30_000)
    tl.push_frame(_active_frame("mic", 10, rms=0.5))
    tl.push_frame(_active_frame("system", 20, rms=0.5))
    assert attribute_word(_word(0, 100), tl, dominance_ratio=4.0) == UNKNOWN_CHANNEL


def test_roll_up_turn_channel_duration_weighted():
    words = [_word(0, 1000), _word(1000, 1100)]
    words[0].channel = "mic"
    words[1].channel = "system"
    # mic spans 1000ms vs system 100ms.
    assert roll_up_turn_channel(words) == "mic"


def test_roll_up_turn_channel_tie_is_unknown():
    words = [_word(0, 100), _word(100, 200)]
    words[0].channel = "mic"
    words[1].channel = "system"
    assert roll_up_turn_channel(words) == UNKNOWN_CHANNEL


def test_attribute_turn_sets_word_and_turn_channels():
    tl = VadTimeline(30_000)
    for ts in (10, 30, 50, 110, 130):
        tl.push_frame(_active_frame("system", ts))
    turn = _turn([_word(0, 100), _word(100, 200)])
    attribute_turn(turn, tl, dominance_ratio=4.0)
    assert turn.words[0].channel == "system"
    assert turn.words[1].channel == "system"
    assert turn.channel == "system"


# --------------------------------------------------------------------------- #
# unknown-channel resolution
# --------------------------------------------------------------------------- #
def test_resolve_unknown_by_window_fills_from_neighbors():
    turn = _turn([_word(0, 50), _word(60, 110), _word(120, 170)])
    turn.words[0].channel = "mic"
    turn.words[1].channel = UNKNOWN_CHANNEL
    turn.words[2].channel = "mic"
    resolve_unknown_channels_by_window(turn, resolution_window_words=2)
    assert turn.words[1].channel == "mic"
    assert turn.words[1].channel_resolved is True
    assert turn.channel == "mic"


def test_resolve_unknown_by_window_leaves_isolated_unknown():
    turn = _turn([_word(0, 50)])
    turn.words[0].channel = UNKNOWN_CHANNEL
    resolve_unknown_channels_by_window(turn, resolution_window_words=2)
    assert turn.words[0].channel == UNKNOWN_CHANNEL


def test_resolve_unknown_by_speaker_history():
    tl = VadTimeline(30_000)
    # Speaker A clearly on mic during 0-100ms.
    for ts in (10, 30, 50, 70):
        tl.push_frame(_active_frame("mic", ts, rms=1.0))
    history = {}
    # First turn: word resolves to mic via VAD, building speaker A's history.
    turn1 = _turn([_word(0, 100, speaker="A")])
    attribute_turn(turn1, tl, dominance_ratio=4.0)
    resolve_unknown_channels_by_speaker_history(
        turn1, tl, history, min_rms_evidence=0.5, dominance_ratio=3.0
    )
    assert turn1.words[0].channel == "mic"

    # Second turn: same speaker, but no VAD frames in the word window -> unknown,
    # then back-filled from accumulated speaker history.
    turn2 = _turn([_word(5000, 5100, speaker="A")])
    attribute_turn(turn2, tl, dominance_ratio=4.0)
    assert turn2.words[0].channel == UNKNOWN_CHANNEL
    resolve_unknown_channels_by_speaker_history(
        turn2, tl, history, min_rms_evidence=0.5, dominance_ratio=3.0
    )
    assert turn2.words[0].channel == "mic"
    assert turn2.words[0].channel_resolved is True


# --------------------------------------------------------------------------- #
# mixer
# --------------------------------------------------------------------------- #
def _mixer(channels=("mic", "system")) -> _ChannelMixer:
    return _ChannelMixer(list(channels), SAMPLE_RATE, ChannelAttributionOptions())


def test_mixer_averages_channels_to_mono():
    mixer = _mixer()
    n = 800  # 50ms @ 16kHz -> meets the min-chunk floor
    mixer.ingest("mic", _pcm(1000, n))
    mixer.ingest("system", _pcm(2000, n))
    chunks = mixer.drain()
    assert len(chunks) == 1
    out = array("h")
    out.frombytes(chunks[0])
    assert len(out) == n
    assert all(s == 1500 for s in out)  # round((1000 + 2000) / 2)


def test_mixer_gates_on_shorter_buffer():
    mixer = _mixer()
    mixer.ingest("mic", _pcm(1000, 1600))  # 100ms
    mixer.ingest("system", _pcm(2000, 800))  # 50ms
    chunks = mixer.drain()
    # Only the 800 overlapping samples can be mixed; the rest stays buffered.
    assert sum(len(b) // 2 for b in chunks) == 800


def test_mixer_min_chunk_floor_holds_small_buffers():
    mixer = _mixer()
    mixer.ingest("mic", _pcm(1000, 100))  # < 50ms
    mixer.ingest("system", _pcm(2000, 100))
    assert mixer.drain() == []  # below the floor
    # force=True bypasses the floor (final flush).
    assert len(mixer.drain(force=True)) == 1


def test_mixer_caps_chunks_at_max_duration():
    mixer = _mixer()
    n = 16000  # 1 second -> must split into >=5 chunks of <=200ms (3200 samples)
    mixer.ingest("mic", _pcm(1000, n))
    mixer.ingest("system", _pcm(1000, n))
    chunks = mixer.drain()
    assert len(chunks) >= 5
    assert all(len(b) // 2 <= 3200 for b in chunks)
    assert sum(len(b) // 2 for b in chunks) == n


def test_mixer_rejects_odd_length_pcm():
    mixer = _mixer()
    with pytest.raises(ValueError):
        mixer.ingest("mic", b"\x00\x01\x02")


def test_mixer_force_flush_pads_missing_channel():
    # Regression: a channel that produced nothing must not gate the flush and
    # swallow the other channel's audio.
    mixer = _mixer()
    mixer.ingest("mic", _pcm(1000, 800))  # system gets nothing
    chunks = mixer.drain(force=True)
    assert sum(len(b) // 2 for b in chunks) == 800  # mic's audio still sent


def test_mixer_close_channel_lets_survivor_drain():
    mixer = _mixer()
    mixer.ingest("mic", _pcm(1000, 1600))  # 100ms
    mixer.ingest("system", _pcm(2000, 800))  # 50ms
    assert sum(len(b) // 2 for b in mixer.drain()) == 800  # aligned overlap only
    # system's source ends -> the remaining mic audio drains without waiting.
    mixer.close_channel("system")
    assert sum(len(b) // 2 for b in mixer.drain()) == 800


# --------------------------------------------------------------------------- #
# coordinators
# --------------------------------------------------------------------------- #
class _FakeSyncClient:
    def __init__(self):
        self._handlers = {event: [] for event in StreamingEvents}
        self.sent = []

    def on(self, event, handler):
        self._handlers[event].append(handler)

    def stream(self, data):
        self.sent.append(data)

    def dispatch_turn(self, turn):
        for handler in self._handlers[StreamingEvents.Turn]:
            handler(self, turn)


class _FakeAsyncClient(_FakeSyncClient):
    async def stream(self, data):  # type: ignore[override]
        self.sent.append(data)


def test_channel_streamer_validates_channels():
    client = _FakeSyncClient()
    with pytest.raises(ValueError):
        ChannelStreamer(client, channels=["mic"], sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError):
        ChannelStreamer(client, channels=["mic", "mic"], sample_rate=SAMPLE_RATE)


def test_channel_streamer_unknown_channel_raises():
    client = _FakeSyncClient()
    mixer = ChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError):
        mixer.stream("speaker", _pcm(1, 800))


def test_invalid_resolve_method_raises_at_construction():
    # Must fail fast at construction, not silently inside the swallowed handler.
    with pytest.raises(ValueError):
        ChannelAttributionOptions(resolve_unknown_channels_method="bogus")


def test_channel_streamer_close_channel_drains_survivor():
    client = _FakeSyncClient()
    mixer = ChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)
    mixer.stream("mic", _pcm(1000, 1600))  # 100ms
    mixer.stream("system", _pcm(2000, 800))  # 50ms
    before = sum(len(b) // 2 for b in client.sent)
    mixer.close_channel("system")  # system's source ended
    after = sum(len(b) // 2 for b in client.sent)
    assert after > before  # surviving mic audio flushed instead of stalling


def test_channel_streamer_sends_mixed_mono():
    client = _FakeSyncClient()
    mixer = ChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)
    mixer.stream("mic", _pcm(1000, 800))
    mixer.stream("system", _pcm(2000, 800))
    assert client.sent, "expected a mixed mono chunk to be sent"
    out = array("h")
    out.frombytes(b"".join(client.sent))
    assert all(s == 1500 for s in out)


def test_channel_streamer_enriches_turn_on_mixer_handler():
    client = _FakeSyncClient()
    mixer = ChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)

    seen = {}

    def on_turn(_client, turn):
        # Handler receives the enriched DualChannelTurnEvent.
        seen["type"] = type(turn).__name__
        seen["channel"] = turn.channel
        seen["word_channel"] = turn.words[0].channel

    mixer.on(StreamingEvents.Turn, on_turn)

    # Drive enough mic audio to register active VAD frames in the word window.
    mixer.stream("mic", _pcm(8000, 1600))
    mixer.stream("system", _pcm(0, 1600))

    # The client emits a base TurnEvent; the mixer enriches + dispatches it.
    client.dispatch_turn(_turn([_word(0, 80)]))
    assert seen["type"] == "DualChannelTurnEvent"
    assert seen["channel"] == "mic"
    assert seen["word_channel"] == "mic"


def test_channel_streamer_none_method_leaves_unknown():
    client = _FakeSyncClient()
    opts = ChannelAttributionOptions(resolve_unknown_channels_method="none")
    mixer = ChannelStreamer(
        client, ["mic", "system"], sample_rate=SAMPLE_RATE, attribution=opts
    )
    captured = {}
    mixer.on(StreamingEvents.Turn, lambda c, t: captured.update(ch=t.words[0].channel))
    # No audio ingested -> no VAD -> word is unknown and stays unknown.
    client.dispatch_turn(_turn([_word(0, 80)]))
    assert captured["ch"] == UNKNOWN_CHANNEL


def test_channel_streamer_forwards_non_turn_events_to_client():
    client = _FakeSyncClient()
    mixer = ChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)

    def on_error(_c, _e):
        pass

    mixer.on(StreamingEvents.Error, on_error)
    # Non-Turn handlers are registered straight through on the underlying client.
    assert on_error in client._handlers[StreamingEvents.Error]


def test_channel_streamer_invokes_vad_callback():
    client = _FakeSyncClient()
    frames = []
    mixer = ChannelStreamer(
        client,
        ["mic", "system"],
        sample_rate=SAMPLE_RATE,
        on_vad=frames.append,
    )
    mixer.stream("mic", _pcm(5000, 320))  # exactly one 20ms VAD frame
    assert len(frames) == 1
    assert frames[0].channel == "mic"


@pytest.mark.asyncio
async def test_async_channel_streamer_sends_mixed_mono():
    client = _FakeAsyncClient()
    mixer = AsyncChannelStreamer(client, ["mic", "system"], sample_rate=SAMPLE_RATE)
    await mixer.stream("mic", _pcm(1000, 800))
    await mixer.stream("system", _pcm(2000, 800))
    await mixer.flush()
    assert client.sent
    out = array("h")
    out.frombytes(b"".join(client.sent))
    assert all(s == 1500 for s in out)
