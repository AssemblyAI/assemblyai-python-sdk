import math
from typing import Any, Dict, Iterator, Optional

import httpx

from .models import TTSError, TTSResponse, TTSResponseInfo

# The one route this client posts to. Streamed and buffered responses differ
# only in the body's `stream` field.
ENDPOINT_SPEECH = "/v1/speech"

# Statuses an answer made before the request reached the API can carry; those
# answers have no JSON body, so `retryable` is decided from the status alone.
_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

_CUT_MESSAGE = "The response was cut before the audio was complete. Retry the request."


def _seconds(value: Any) -> Optional[int]:
    """A retry hint as whole seconds, rounded up; `None` for anything else."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(math.ceil(value))
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _int_header(headers: httpx.Headers, name: str) -> Optional[int]:
    return _seconds(headers.get(name))


def _header_fields(headers: httpx.Headers) -> Dict[str, Any]:
    """The `TTSResponseInfo` fields read from the `X-TTS-*` headers of a `200`."""

    return {
        "request_id": headers.get("x-request-id"),
        "model": headers.get("x-tts-model"),
        "voice": headers.get("x-tts-voice"),
        "language": headers.get("x-tts-language"),
        "encoding": headers.get("x-tts-encoding"),
        "sample_rate": _int_header(headers, "x-tts-sample-rate"),
        "container": headers.get("x-tts-container"),
        "bit_rate": _int_header(headers, "x-tts-bit-rate"),
        "content_type": headers.get("content-type"),
        "voice_expires_at": _int_header(headers, "x-tts-voice-expires-at"),
        "billed_characters": _int_header(headers, "x-tts-billed-characters"),
        "audio_duration_ms": _int_header(headers, "x-tts-audio-duration-ms"),
    }


def _info_from_headers(headers: httpx.Headers) -> TTSResponseInfo:
    return TTSResponseInfo(**_header_fields(headers))


def _error_from_response(response: httpx.Response) -> TTSError:
    """
    Builds a `TTSError` from a non-200 response.

    The API answers every refusal with one JSON envelope (`code`, `message`,
    `param`, `request_id`, `errors`, `retryable`, `retry_after_seconds`).
    An answer made before the request reached the API (a load balancer's
    `502`, a plain-text `404`) has no envelope: its status and text are
    reported, and `retryable` follows the status.
    """

    status = response.status_code
    retry_after = _int_header(response.headers, "retry-after")
    request_id = response.headers.get("x-request-id")

    body: Any = None
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            body = response.json()
        except ValueError:
            body = None

    if isinstance(body, dict):
        message = body.get("message") or body.get("error")
        if not message:
            message = f"text-to-speech request failed with status {status}"
        if body.get("retry_after_seconds") is not None:
            retry_after = _seconds(body["retry_after_seconds"])
        retryable = body.get("retryable")
        if retryable is None:
            retryable = status in _RETRYABLE_STATUSES
        return TTSError(
            message,
            status_code=status,
            error_code=body.get("code"),
            param=body.get("param"),
            request_id=body.get("request_id") or request_id,
            retryable=retryable,
            retry_after=retry_after,
            errors=body.get("errors"),
        )

    text = response.text.strip()
    message = f"text-to-speech request failed with status {status}"
    if text:
        message = f"{message}: {text[:200]}"
    return TTSError(
        message,
        status_code=status,
        request_id=request_id,
        retryable=status in _RETRYABLE_STATUSES,
        retry_after=retry_after,
    )


def _cut_error(info: Optional[TTSResponseInfo]) -> TTSError:
    """The error for a body cut after the first audio: retry the request."""

    return TTSError(
        _CUT_MESSAGE,
        status_code=200,
        request_id=info.request_id if info else None,
        retryable=True,
    )


class TTSStream:
    """
    The audio of one request, read as the API generates it.

    Returned by `TextToSpeech.stream()`. The request is sent when the stream
    is entered as a context manager or first read, and `TTSError` is raised
    there if the API refuses it. Use `with`, or call `close()`, so the
    connection is released: a fully read stream's connection returns to the
    pool, a partially read one is dropped. `info` holds what the response
    headers say (model, voice, encoding, sample rate, request id) once the
    request has been sent; `iter_bytes()` yields the audio as it arrives and
    `read()` collects all of it.

    The API commits to `200` with the first audio, so a failure after that
    cuts the body instead of changing the status: the iteration then raises
    `TTSError` with `retryable=True`. Retry the whole request. Chunks are
    not aligned to samples: for 16-bit PCM, carry an odd trailing byte over
    to the next chunk.
    """

    def __init__(self, client: httpx.Client, request: httpx.Request) -> None:
        self._client = client
        self._request = request
        self._response: Optional[httpx.Response] = None
        self._info: Optional[TTSResponseInfo] = None

    @property
    def info(self) -> Optional[TTSResponseInfo]:
        """The response headers; `None` until the request has been sent."""

        return self._info

    def _open(self) -> httpx.Response:
        if self._response is not None:
            return self._response

        response = self._client.send(self._request, stream=True)
        if response.status_code != httpx.codes.OK:
            try:
                response.read()
                raise _error_from_response(response)
            finally:
                response.close()

        self._response = response
        self._info = _info_from_headers(response.headers)
        return response

    def __enter__(self) -> "TTSStream":
        self._open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[bytes]:
        return self.iter_bytes()

    def iter_bytes(self) -> Iterator[bytes]:
        """Yields the audio bytes as they arrive."""

        response = self._open()
        try:
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk
        except httpx.RemoteProtocolError as exc:
            raise _cut_error(self._info) from exc

    def read(self) -> bytes:
        """Reads the rest of the audio and returns it."""

        return b"".join(self.iter_bytes())

    def close(self) -> None:
        """Releases the connection. Audio not yet read is discarded."""

        if self._response is not None:
            self._response.close()


def synthesize(
    client: httpx.Client,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
) -> TTSResponse:
    """
    Posts a buffered speech request and returns the complete audio.

    Asks for `"stream": false`, so nothing arrives until the audio is
    complete and an API failure is always an error status, never a cut body.

    Args:
        client: the HTTP client (carries the `Authorization` header).
        base_url: the Text-to-Speech API base URL, e.g. `https://tts.us.assemblyai.com`.
        body: the JSON body: `text` and the options of a `TTSConfig`.
        timeout: the httpx timeout of each socket operation.

    Raises: `TTSError` on any non-200 response.
    """

    response = client.post(
        base_url.rstrip("/") + ENDPOINT_SPEECH,
        json={**body, "stream": False},
        timeout=timeout,
    )

    if response.status_code != httpx.codes.OK:
        raise _error_from_response(response)

    return TTSResponse(audio=response.content, **_header_fields(response.headers))


def stream_speech(
    client: httpx.Client,
    *,
    base_url: str,
    body: Dict[str, Any],
    timeout: float,
) -> TTSStream:
    """
    Prepares a streamed speech request and returns it as a `TTSStream`.

    Nothing is sent here: the stream sends the request when it is entered or
    first read, and raises `TTSError` then if the API refuses it.
    """

    request = client.build_request(
        "POST",
        base_url.rstrip("/") + ENDPOINT_SPEECH,
        json={**body, "stream": True},
        timeout=timeout,
    )
    return TTSStream(client, request)
