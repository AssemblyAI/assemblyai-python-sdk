"""Client-side dual / multi-channel support for streaming v3.

Note-taker use case: capture two live sources (microphone + system audio) as
one streaming session while still knowing which physical source each word came
from.

- Each named channel's PCM is fed in separately via ``ChannelStreamer.stream``.
- Per-channel energy VAD records which channel was acoustically active when.
- The channels are summed into one mono stream sent over the existing single
  websocket session.
- On every ``Turn``, words are attributed back to a channel by matching their
  server timestamps against the per-channel VAD timeline; the enriched
  ``TurnEvent`` (``turn.channel`` + per-word ``word.channel``) is delivered to
  the handler registered on the coordinator.

Attribution is purely client-side, so any ``speech_model`` works unchanged and
channel (physical source) stays independent of diarization ``speaker``.
"""

import inspect
import logging
import math
import sys
import threading
from array import array
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

from .models import StreamingEvents, TurnEvent, Word

if TYPE_CHECKING:  # avoid an import cycle; only used for type hints
    from .async_client import AsyncStreamingClient
    from .client import StreamingClient

logger = logging.getLogger(__name__)

UNKNOWN_CHANNEL = "unknown"

# 20 ms VAD frames, and the server's per-message audio duration limits.
_VAD_FRAME_MS = 20
_MIN_CHUNK_MS = 50
_MAX_CHUNK_MS = 200


# Enriched event types subclass the base streaming models and add the
# client-side channel fields, so the base ``Word`` / ``TurnEvent`` payloads stay
# clean for single-stream users. A coordinator builds these from the base events
# and delivers them to handlers registered on the coordinator.
class DualChannelWord(Word):
    """A ``Word`` enriched with channel attribution (independent of ``speaker``)."""

    # Physical input channel (e.g. "mic" / "system"), or "unknown" if no channel
    # was clearly dominant during the word's window.
    channel: Optional[str] = None
    # True when ``channel`` was inferred by an unknown-resolution strategy rather
    # than measured directly by VAD.
    channel_resolved: Optional[bool] = None


class DualChannelTurnEvent(TurnEvent):
    """A ``TurnEvent`` enriched with per-word and turn-level channel attribution."""

    words: List[DualChannelWord] = []
    # Duration-weighted majority channel across ``words``, or "unknown".
    channel: Optional[str] = None


@dataclass
class VadResult:
    """One frame's voice-activity decision: whether speech is present and the
    frame's RMS energy (used to weight per-channel attribution)."""

    active: bool
    energy: float


@dataclass
class VadFrame:
    """A per-channel, per-frame VAD observation.

    ``ts`` is stream-relative milliseconds from the channel's own sample counter
    — the same reference frame as ``Word.start`` / ``Word.end``.
    """

    ts: float
    channel: str
    active: bool
    rms: float


class VadDetector:
    """Pluggable per-channel voice-activity detector.

    A separate instance is held per channel. ``process`` receives a fixed-size
    frame of float samples in ``[-1.0, 1.0]`` at the session's sample rate.
    Subclass / duck-type this to drop in a DNN-backed detector for noisy
    environments.
    """

    def process(self, frame: Sequence[float]) -> VadResult:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError


class EnergyVad(VadDetector):
    """Energy-based VAD with adaptive noise-floor tracking and hangover.

    Pure Python, no dependencies. Suited to "which physical channel is speaking"
    since the channels are already physically separated at capture; hand the
    harder speech-vs-noise problem to a custom ``VadDetector`` via
    ``ChannelAttributionOptions.create_vad``.

    Tuning: ``threshold_ratio`` below 2 is over-sensitive, above 6 misses quiet
    onsets/offsets; ``noise_floor_alpha`` above 0.1 adapts to non-stationary
    background faster but risks creeping up onto a sustained quiet voice.
    """

    def __init__(
        self,
        threshold_ratio: float = 3.0,
        noise_floor_alpha: float = 0.05,
        hangover_frames: int = 10,
        initial_noise_floor: float = 1e-4,
    ):
        self._threshold_ratio = threshold_ratio
        self._noise_floor_alpha = noise_floor_alpha
        self._hangover_frames = hangover_frames
        self._initial_noise_floor = initial_noise_floor
        self._noise_floor = initial_noise_floor
        self._hangover_remaining = 0

    def process(self, frame: Sequence[float]) -> VadResult:
        n = len(frame)
        sum_sq = 0.0
        for s in frame:
            sum_sq += s * s
        rms = math.sqrt(sum_sq / n) if n > 0 else 0.0

        threshold = self._noise_floor * self._threshold_ratio
        active = rms > threshold

        if active:
            self._hangover_remaining = self._hangover_frames
        elif self._hangover_remaining > 0:
            self._hangover_remaining -= 1
            active = True
            # In hangover, don't update the floor — RMS may still be tail energy.
        else:
            self._noise_floor = (
                self._noise_floor * (1 - self._noise_floor_alpha)
                + rms * self._noise_floor_alpha
            )

        return VadResult(active=active, energy=rms)

    def reset(self) -> None:
        self._noise_floor = self._initial_noise_floor
        self._hangover_remaining = 0


class VadTimeline:
    """Append-only ring buffer of ``VadFrame``s in stream-relative ms order.

    ``push_frame`` is amortized O(1); ``frames_in_window`` is O(n) over kept
    frames, fine for the per-word lookups done here.
    """

    def __init__(self, window_ms: int):
        self._window_ms = window_ms
        self._frames: List[VadFrame] = []
        self._head = 0
        # The threaded ``StreamingClient`` runs ``frames_in_window`` on the read
        # thread while the user thread runs ``push_frame``; compaction swaps
        # ``_frames`` / ``_head`` non-atomically. The lock keeps push/read/compact
        # mutually exclusive. Uncontended on the async / single-threaded paths.
        self._lock = threading.Lock()

    def push_frame(self, frame: VadFrame) -> None:
        with self._lock:
            self._frames.append(frame)
            cutoff = frame.ts - self._window_ms
            while (
                self._head < len(self._frames) and self._frames[self._head].ts < cutoff
            ):
                self._head += 1
            # Compact occasionally so the list doesn't grow without bound.
            if self._head > 1024 and self._head * 2 > len(self._frames):
                self._frames = self._frames[self._head :]
                self._head = 0

    def frames_in_window(self, start_ms: float, end_ms: float) -> List[VadFrame]:
        out: List[VadFrame] = []
        with self._lock:
            for i in range(self._head, len(self._frames)):
                f = self._frames[i]
                if f.ts < start_ms:
                    continue
                if f.ts > end_ms:
                    break
                out.append(f)
        return out

    def clear(self) -> None:
        with self._lock:
            self._frames = []
            self._head = 0


def _score_channels(frames: Iterable[VadFrame]) -> Dict[str, float]:
    """Sum active-frame RMS per channel. Channels with no active energy are
    omitted from the result."""
    scores: Dict[str, float] = {}
    for f in frames:
        if not f.active:
            continue
        scores[f.channel] = scores.get(f.channel, 0.0) + f.rms
    return scores


def _top_by_ratio(scores: Dict[str, float], dominance_ratio: float) -> Optional[str]:
    """Winner of a per-channel score map: the sole channel if only one had
    energy, else the top channel iff it beats the runner-up by
    ``dominance_ratio``. ``None`` when there's no clear winner (tie / too close)
    or no scores.

    The ratio is a real knob — raising it yields more ``None`` (i.e. "unknown",
    which a resolution strategy may back-fill). This diverges from the Node
    reference, whose absolute-winner fallback makes the equivalent ratio a no-op.
    """
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0]
    (top_name, top_score), (_, runner_score) = ranked[0], ranked[1]
    if runner_score <= 0 or top_score >= dominance_ratio * runner_score:
        return top_name
    return None


def attribute_word(
    word: Word,
    timeline: VadTimeline,
    dominance_ratio: float,
) -> str:
    """Channel that dominated the word's ``[start, end]`` window, or "unknown"
    if no channel had energy there or none was clearly dominant."""
    scores = _score_channels(timeline.frames_in_window(word.start, word.end))
    winner = _top_by_ratio(scores, dominance_ratio)
    return winner if winner is not None else UNKNOWN_CHANNEL


def roll_up_turn_channel(words: Sequence[DualChannelWord]) -> str:
    """Duration-weighted majority of per-word channels. "unknown" if there are
    no resolved words or two channels tie exactly."""
    totals: Dict[str, float] = {}
    for w in words:
        if not w.channel or w.channel == UNKNOWN_CHANNEL:
            continue
        dur = max(0, w.end - w.start)
        totals[w.channel] = totals.get(w.channel, 0.0) + dur
    if not totals:
        return UNKNOWN_CHANNEL
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0]
    (top_name, top_ms), (_, runner_ms) = ranked[0], ranked[1]
    if top_ms == runner_ms:
        return UNKNOWN_CHANNEL
    return top_name


def attribute_turn(
    turn: DualChannelTurnEvent,
    timeline: VadTimeline,
    dominance_ratio: float,
) -> None:
    """Write ``turn.words[i].channel`` for every word and set ``turn.channel``
    to the duration-weighted rollup. Mutates the enriched ``turn`` in place."""
    for w in turn.words:
        w.channel = attribute_word(w, timeline, dominance_ratio)
    turn.channel = roll_up_turn_channel(turn.words)


def resolve_unknown_channels_by_window(
    turn: DualChannelTurnEvent,
    resolution_window_words: int,
) -> None:
    """Back-fill "unknown" words from the dominant non-"unknown" channel among
    +/-N neighbor words in the same turn. Words with no resolved neighbors stay
    "unknown"; confident decisions are never modified; resolved words are tagged
    ``channel_resolved = True``."""
    words = turn.words
    mutated = False
    for i, target in enumerate(words):
        if target.channel != UNKNOWN_CHANNEL:
            continue
        tally: Dict[str, int] = {}
        lo = max(0, i - resolution_window_words)
        hi = min(len(words) - 1, i + resolution_window_words)
        for j in range(lo, hi + 1):
            if j == i:
                continue
            ch = words[j].channel
            if not ch or ch == UNKNOWN_CHANNEL:
                continue
            tally[ch] = tally.get(ch, 0) + 1
        if not tally:
            continue
        top: Optional[str] = None
        top_count = 0
        tied = False
        for name, count in tally.items():
            if count > top_count:
                top, top_count, tied = name, count, False
            elif count == top_count:
                tied = True
        if top is not None and not tied:
            target.channel = top
            target.channel_resolved = True
            mutated = True
    if mutated:
        turn.channel = roll_up_turn_channel(words)


def resolve_unknown_channels_by_speaker_history(
    turn: DualChannelTurnEvent,
    timeline: VadTimeline,
    speaker_history: Dict[str, Dict[str, float]],
    min_rms_evidence: float,
    dominance_ratio: float,
) -> None:
    """Back-fill "unknown" words using each speaker's session-wide channel
    evidence: per speaker, sum active VAD-frame RMS per channel across all their
    words so far (accumulated into ``speaker_history``). A speaker is resolvable
    once their total evidence clears ``min_rms_evidence`` and their top channel
    beats the runner-up by ``dominance_ratio``. Only touches "unknown" words;
    ``speaker`` is never modified."""
    # 1. Accumulate evidence from this turn's words.
    for w in turn.words:
        if not w.speaker:
            continue
        entry = speaker_history.setdefault(w.speaker, {})
        for f in timeline.frames_in_window(w.start, w.end):
            if not f.active:
                continue
            entry[f.channel] = entry.get(f.channel, 0.0) + f.rms

    # 2. Fill unknown words whose speakers have dominant evidence.
    mutated = False
    for w in turn.words:
        if w.channel != UNKNOWN_CHANNEL or not w.speaker:
            continue
        entry = speaker_history.get(w.speaker)
        if not entry or sum(entry.values()) < min_rms_evidence:
            continue
        winner = _top_by_ratio(entry, dominance_ratio)
        if winner is not None:
            w.channel = winner
            w.channel_resolved = True
            mutated = True
    if mutated:
        turn.channel = roll_up_turn_channel(turn.words)


RESOLVE_UNKNOWN_METHODS = ("none", "window", "speaker-history")


@dataclass
class ChannelAttributionOptions:
    """Tuning for client-side channel attribution. All fields have sane defaults
    matching the Node SDK; override only when needed."""

    # Per-word energy ratio above which a channel is declared dominant.
    dominance_ratio: float = 4.0
    # How far back the VAD timeline retains frames for per-word lookups.
    timeline_window_ms: int = 30_000
    # Factory for the per-channel detector; called once per channel with its
    # name. Defaults to a fresh ``EnergyVad`` per channel.
    create_vad: Optional[Callable[[str], VadDetector]] = None
    # How to fill words VAD couldn't attribute: "none" | "window" |
    # "speaker-history".
    resolve_unknown_channels_method: str = "window"
    # "window": +/-N neighbor words consulted to fill an unknown word.
    resolution_window_words: int = 2
    # "speaker-history": minimum cumulative active-RMS before a speaker's
    # channel is considered established.
    speaker_history_min_rms_evidence: float = 0.5
    # "speaker-history": top channel must beat the runner-up by this ratio.
    speaker_history_dominance_ratio: float = 3.0

    def __post_init__(self) -> None:
        # Fail fast at construction; an invalid method would otherwise be
        # swallowed inside the Turn handler, silently disabling resolution.
        if self.resolve_unknown_channels_method not in RESOLVE_UNKNOWN_METHODS:
            raise ValueError(
                "resolve_unknown_channels_method must be one of "
                f"{RESOLVE_UNKNOWN_METHODS}; got "
                f"{self.resolve_unknown_channels_method!r}."
            )

    def _make_vad(self, channel: str) -> VadDetector:
        if self.create_vad is not None:
            return self.create_vad(channel)
        return EnergyVad()


_BIG_ENDIAN = sys.byteorder == "big"


def _pcm16_to_array(data: bytes) -> array:
    """Parse little-endian 16-bit PCM bytes into a signed-short ``array``."""
    if len(data) % 2 != 0:
        raise ValueError(
            f"PCM data length must be even (16-bit samples); got {len(data)} bytes."
        )
    samples = array("h")
    samples.frombytes(bytes(data))
    if _BIG_ENDIAN:
        samples.byteswap()  # interpret the bytes as little-endian
    return samples


def _array_to_pcm16(samples: array) -> bytes:
    """Serialize a signed-short ``array`` back to little-endian 16-bit PCM."""
    if _BIG_ENDIAN:
        samples = samples[:]
        samples.byteswap()
    return samples.tobytes()


class _ChannelMixer:
    """Owns per-channel PCM buffers, per-channel VAD, the shared VAD timeline,
    and the mono mixing math. Runtime-agnostic — both coordinators drive it."""

    def __init__(
        self,
        channels: Sequence[str],
        sample_rate: int,
        options: ChannelAttributionOptions,
    ):
        self.channels = list(channels)
        self.sample_rate = sample_rate
        self.timeline = VadTimeline(options.timeline_window_ms)

        self._vad_frame_samples = max(1, round(sample_rate * _VAD_FRAME_MS / 1000))
        self._min_chunk_samples = max(1, round(sample_rate * _MIN_CHUNK_MS / 1000))
        self._max_chunk_samples = max(
            self._min_chunk_samples, round(sample_rate * _MAX_CHUNK_MS / 1000)
        )

        self._buffers: Dict[str, array] = {n: array("h") for n in self.channels}
        self._vad_frame: Dict[str, List[float]] = {n: [] for n in self.channels}
        self._received: Dict[str, int] = {n: 0 for n in self.channels}
        self._vads: Dict[str, VadDetector] = {
            n: options._make_vad(n) for n in self.channels
        }
        # Channels closed via ``close_channel`` — treated as silence so their
        # absence no longer gates mixing for the survivors.
        self._ended: Set[str] = set()

    def close_channel(self, channel: str) -> None:
        """Mark a channel finished (source ended). Subsequent ``drain`` calls
        stop waiting on it and pad it with silence."""
        self._ended.add(channel)

    def ingest(
        self,
        channel: str,
        data: bytes,
        on_vad: Optional[Callable[[VadFrame], None]] = None,
    ) -> None:
        samples = _pcm16_to_array(data)
        self._buffers[channel].extend(samples)

        vad = self._vads[channel]
        frame_buf = self._vad_frame[channel]
        received = self._received[channel]
        for s in samples:
            frame_buf.append(s / 0x8000)
            received += 1
            if len(frame_buf) == self._vad_frame_samples:
                result = vad.process(frame_buf)
                frame = VadFrame(
                    ts=received / self.sample_rate * 1000,
                    channel=channel,
                    active=result.active,
                    rms=result.energy,
                )
                self.timeline.push_frame(frame)
                if on_vad is not None:
                    on_vad(frame)
                frame_buf.clear()
        self._received[channel] = received

    def drain(self, force: bool = False) -> List[bytes]:
        """Mix buffered audio into mono PCM chunks, each clamped to
        ``[_MIN_CHUNK_MS, _MAX_CHUNK_MS]`` (the ``_MIN_CHUNK_MS`` floor applies
        only while not ``force``).

        While every channel is still feeding, mixing gates on the shortest live
        buffer to keep channels time-aligned. Once a channel is closed
        (``close_channel``) or on the final ``force`` flush, shorter/ended
        buffers are zero-padded up to the longest instead of gating on them, so
        a terminated source degrades to the survivors rather than stalling the
        session and dropping everything accumulated since.
        """
        bufs = [self._buffers[n] for n in self.channels]
        divisor = len(bufs)
        # Pad (don't gate) once any channel has ended, or on the final flush.
        pad = force or bool(self._ended)
        out_chunks: List[bytes] = []
        while True:
            if pad:
                mix_len = max((len(b) for b in bufs), default=0)
            else:
                live = [
                    len(self._buffers[n]) for n in self.channels if n not in self._ended
                ]
                mix_len = min(live) if live else 0
            if mix_len == 0:
                break
            if not force and mix_len < self._min_chunk_samples:
                break
            if mix_len > self._max_chunk_samples:
                mix_len = self._max_chunk_samples
            out = array("h", bytes(2 * mix_len))
            for i in range(mix_len):
                total = 0
                for b in bufs:
                    total += b[i] if i < len(b) else 0  # pad past buffer end
                avg = round(total / divisor)
                out[i] = -32768 if avg < -32768 else (32767 if avg > 32767 else avg)
            for b in bufs:
                del b[: min(mix_len, len(b))]
            out_chunks.append(_array_to_pcm16(out))
        return out_chunks


def _validate_channels(channels: Sequence[str]) -> List[str]:
    names = list(channels)
    if len(names) < 2:
        raise ValueError("channels must declare at least 2 channel names.")
    if len(set(names)) != len(names):
        raise ValueError("channels names must be unique.")
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("channel names must be non-empty strings.")
    return names


class _BaseChannelStreamer:
    """Shared dual/multi-channel coordination independent of the wrapped
    client's sync/async I/O. Channel config lives here, never on
    ``StreamingParameters`` (it must not reach the websocket URL); the wrapped
    client streams ordinary mono audio and is otherwise untouched.
    """

    def __init__(
        self,
        channels: Sequence[str],
        sample_rate: int,
        options: Optional[ChannelAttributionOptions],
        on_vad: Optional[Callable[[VadFrame], None]],
    ):
        self.channels = _validate_channels(channels)
        self._options = options or ChannelAttributionOptions()
        self._on_vad = on_vad
        self._mixer = _ChannelMixer(self.channels, sample_rate, self._options)
        self._speaker_history: Dict[str, Dict[str, float]] = {}
        self._turn_handlers: List[Callable] = []
        # Set by each subclass in __init__ (the concrete sync/async client).
        self._client: Union["StreamingClient", "AsyncStreamingClient"]

    def on(self, event: StreamingEvents, handler: Callable) -> None:
        """Register an event handler. ``Turn`` events are delivered as an
        enriched ``DualChannelTurnEvent``; all other events are forwarded to the
        underlying client unchanged."""
        if event == StreamingEvents.Turn:
            self._turn_handlers.append(handler)
        else:
            self._client.on(event, handler)

    def _check_channel(self, channel: str) -> None:
        if channel not in self._mixer._buffers:
            raise ValueError(
                f'Unknown channel "{channel}"; declared channels: '
                f"{', '.join(self.channels)}."
            )

    def _enrich(self, base_turn: TurnEvent) -> DualChannelTurnEvent:
        """Build a ``DualChannelTurnEvent`` from the base turn (left untouched)
        and run channel attribution + the configured unknown-resolution."""
        # Dump the base event keeping None fields (pydantic v1/v2).
        data = (
            base_turn.model_dump()
            if hasattr(base_turn, "model_dump")
            else base_turn.dict()
        )
        enriched = DualChannelTurnEvent(**data)
        attribute_turn(enriched, self._mixer.timeline, self._options.dominance_ratio)
        method = self._options.resolve_unknown_channels_method
        if method == "window":
            resolve_unknown_channels_by_window(
                enriched, self._options.resolution_window_words
            )
        elif method == "speaker-history":
            resolve_unknown_channels_by_speaker_history(
                enriched,
                self._mixer.timeline,
                self._speaker_history,
                self._options.speaker_history_min_rms_evidence,
                self._options.speaker_history_dominance_ratio,
            )
        # method == "none": leave "unknown" words as-is (validated at construction).
        return enriched

    @staticmethod
    def _as_chunks(data: Union[bytes, Iterable[bytes]]) -> Iterable[bytes]:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return [bytes(data)]
        return data


class ChannelStreamer(_BaseChannelStreamer):
    """Dual/multi-channel coordinator for the threaded ``StreamingClient``.

    Feed each named channel's 16-bit little-endian PCM via ``stream(channel,
    data)``; the channels are summed into one mono stream over the client's
    single session. Register handlers on the coordinator (``mixer.on(...)``):
    ``Turn`` handlers receive an enriched ``DualChannelTurnEvent``; all other
    events are forwarded to the wrapped client, keeping the base payloads clean
    for single-stream use.

    Requires ``pcm_s16le`` (linear mixing is invalid for ``pcm_mulaw``). Feed
    every channel continuous PCM at the same ``sample_rate`` (silence as zeros);
    call ``close_channel(name)`` when a source ends mid-session so the session
    degrades to the survivors, and ``flush()`` before
    ``client.disconnect(terminate=True)`` to push the trailing audio.
    """

    def __init__(
        self,
        client: "StreamingClient",
        channels: Sequence[str],
        sample_rate: int,
        attribution: Optional[ChannelAttributionOptions] = None,
        on_vad: Optional[Callable[[VadFrame], None]] = None,
    ):
        super().__init__(channels, sample_rate, attribution, on_vad)
        self._client = client
        client.on(StreamingEvents.Turn, self._handle_turn)

    def _handle_turn(self, client: object, base_turn: TurnEvent) -> None:
        enriched = self._enrich(base_turn)
        for handler in self._turn_handlers:
            try:
                handler(client, enriched)
            except Exception:
                logger.exception("dual-channel on_turn handler raised")

    def stream(self, channel: str, data: Union[bytes, Iterable[bytes]]) -> None:
        """Ingest PCM for ``channel``, then send whatever can now be mixed."""
        self._check_channel(channel)
        for chunk in self._as_chunks(data):
            self._mixer.ingest(channel, chunk, on_vad=self._on_vad)
        for mixed in self._mixer.drain():
            self._client.stream(mixed)

    def close_channel(self, channel: str) -> None:
        """Signal that ``channel``'s source has ended; the session keeps
        streaming the survivors, sending any newly mixable audio immediately."""
        self._check_channel(channel)
        self._mixer.close_channel(channel)
        for mixed in self._mixer.drain():
            self._client.stream(mixed)

    def flush(self) -> None:
        """Mix and send any remaining buffered audio. Call before
        ``client.disconnect``."""
        for mixed in self._mixer.drain(force=True):
            self._client.stream(mixed)


class AsyncChannelStreamer(_BaseChannelStreamer):
    """Asyncio-native counterpart to ``ChannelStreamer`` (wraps
    ``AsyncStreamingClient``); ``stream`` / ``close_channel`` / ``flush`` are
    coroutines. ``Turn`` handlers may be sync or ``async`` (awaited inline on the
    read task). See ``ChannelStreamer`` for requirements.
    """

    def __init__(
        self,
        client: "AsyncStreamingClient",
        channels: Sequence[str],
        sample_rate: int,
        attribution: Optional[ChannelAttributionOptions] = None,
        on_vad: Optional[Callable[[VadFrame], None]] = None,
    ):
        super().__init__(channels, sample_rate, attribution, on_vad)
        self._client = client
        client.on(StreamingEvents.Turn, self._handle_turn)

    async def _handle_turn(self, client: object, base_turn: TurnEvent) -> None:
        enriched = self._enrich(base_turn)
        for handler in self._turn_handlers:
            try:
                result = handler(client, enriched)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("dual-channel on_turn handler raised")

    async def stream(self, channel: str, data: Union[bytes, Iterable[bytes]]) -> None:
        self._check_channel(channel)
        for chunk in self._as_chunks(data):
            self._mixer.ingest(channel, chunk, on_vad=self._on_vad)
        for mixed in self._mixer.drain():
            await self._client.stream(mixed)

    async def close_channel(self, channel: str) -> None:
        """Signal that ``channel``'s source has ended; the session keeps
        streaming the survivors instead of stalling."""
        self._check_channel(channel)
        self._mixer.close_channel(channel)
        for mixed in self._mixer.drain():
            await self._client.stream(mixed)

    async def flush(self) -> None:
        for mixed in self._mixer.drain(force=True):
            await self._client.stream(mixed)
