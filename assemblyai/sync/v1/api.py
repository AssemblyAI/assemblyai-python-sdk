from typing import Iterator, Optional

import httpx

from ... import types
from ._multipart import AudioChunks, StreamingMultipartEncoder, iter_chunks

# Canonical paths since the sync API gained a /v1 prefix (#18103); the
# unprefixed routes remain served for SDK versions that predate it.
#
# `/v1/transcribe/live` is the one endpoint this client posts audio to. It is
# also served at `/v1/transcribe/stream`, the path it shipped under, so an
# older deployment still answers there — but `live` is its name.
ENDPOINT_TRANSCRIBE_LIVE = "/v1/transcribe/live"
# The buffered endpoint. Still served, and still exported through the legacy
# `assemblyai.sync_api` shim, but nothing here requests it.
ENDPOINT_TRANSCRIBE = "/v1/transcribe"
ENDPOINT_WARM = "/v1/warm"
MODEL_HEADER = "X-AAI-Model"


def _error_from_response(response: httpx.Response) -> types.SyncTranscriptError:
    """
    Builds a `SyncTranscriptError` from a non-200 response.

    The service returns an RFC 9457 problem-details envelope
    (`{"status", "title", "detail"}`); `error_code` is the snake_cased
    `title` (e.g. `"Audio Too Large"` -> `audio_too_large`). Older envelopes
    (`{"error_code", "message"}`, `{"detail"}`, and `{"error"}`) are still
    accepted; a bare `error` string carries no `error_code`.
    """
    error_code: Optional[str] = None
    message: Optional[str] = None

    try:
        body = response.json()
        if isinstance(body, dict):
            error_code = body.get("error_code")
            title = body.get("title")
            if error_code is None and isinstance(title, str) and title:
                error_code = title.lower().replace(" ", "_")
            message = body.get("detail") or body.get("message")
            if not message:
                error = body.get("error")
                if isinstance(error, str) and error:
                    message = error
    except Exception:
        message = response.text or None

    if not message:
        message = f"sync transcription failed with status {response.status_code}"

    retry_after_header = response.headers.get("retry-after")
    retry_after = (
        int(retry_after_header)
        if retry_after_header and retry_after_header.isdigit()
        else None
    )

    return types.SyncTranscriptError(
        message,
        status_code=response.status_code,
        error_code=error_code,
        retry_after=retry_after,
    )


def transcribe(
    client: httpx.Client,
    *,
    base_url: str,
    audio: bytes,
    filename: str,
    audio_content_type: str,
    model: str,
    config: Optional[dict],
    timeout: float,
) -> types.SyncTranscriptResponse:
    """
    Posts a transcription request for audio that is already complete.

    Sends it over the same live connection `transcribe_live` uses, as a single
    chunk: audio that already exists is a stream whose bytes are all ready at
    once. There is one request shape in this client, so a complete clip and a
    live capture reach the service the same way, and the server transcribes
    each speech segment as it lands either way.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the sync API base URL, e.g. `https://sync.assemblyai.com`.
        audio: raw audio bytes (WAV container or S16LE PCM).
        filename: name for the audio multipart part.
        audio_content_type: `audio/wav` or `audio/pcm`; selects the decoder.
        model: sent as the `X-AAI-Model` routing header.
        config: the JSON `config` part, or None for an empty one.
        timeout: per-operation timeout in seconds.

    Returns: the parsed transcript response.

    Raises: `SyncTranscriptError` on any non-200 response.
    """
    return transcribe_live(
        client,
        base_url=base_url,
        chunks=(audio,),
        filename=filename,
        audio_content_type=audio_content_type,
        model=model,
        config=config,
        timeout=timeout,
    )


def transcribe_live(
    client: httpx.Client,
    *,
    base_url: str,
    chunks: AudioChunks,
    filename: str,
    audio_content_type: str,
    model: str,
    config: Optional[dict],
    timeout: float,
) -> types.SyncTranscriptResponse:
    """
    Posts a transcription request whose audio is uploaded as it arrives.

    Sends the body with chunked transfer encoding — httpx frames an unsized
    iterator that way — so the request can start before the audio exists. The
    server transcribes each speech segment as it lands, leaving only the final
    segment's inference to wait on once the caller stops speaking.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the sync API base URL, e.g. `https://sync.assemblyai.com`.
        chunks: audio pieces (WAV container or S16LE PCM), in order.
        filename: name for the audio multipart part.
        audio_content_type: `audio/wav` or `audio/pcm`; selects the decoder.
        model: sent as the `X-AAI-Model` routing header.
        config: the JSON `config` part. None sends an empty object: the
            streaming endpoint requires the part ahead of the audio.
        timeout: per-operation timeout in seconds, as for every httpx request:
            it bounds connecting, each socket write and each read while waiting
            for the response, not the request end to end. Time blocked in the
            caller's producer is not counted, so it need not cover the
            recording.

    Returns: the parsed transcript response.

    Raises: `SyncTranscriptError` on any non-200 response — including one the
        server sends while the upload is still in flight, which it may do for
        auth, rate-limit and capacity failures. `TypeError` if the producer
        yields something other than bytes. Any other exception the producer
        raises propagates unchanged; the connection is dropped and no
        transcript is returned.
    """
    encoder = StreamingMultipartEncoder()

    def body() -> Iterator[bytes]:
        # config first: the server needs sample_rate and channels before it can
        # decode a single audio byte, and rejects audio that arrives first.
        head = encoder.config_part(config) + encoder.audio_header(
            filename, audio_content_type
        )
        yield head

        for chunk in iter_chunks(chunks):
            if chunk:
                yield chunk

        yield encoder.closing()

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_TRANSCRIBE_LIVE,
        content=body(),
        headers={MODEL_HEADER: model, "Content-Type": encoder.content_type},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return types.SyncTranscriptResponse.parse_obj(response.json())
