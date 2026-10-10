"""Talking to a served model: the two HTTP calls P1 needs to stop being offline.

    an attempt        <- POST {base}/v1/chat/completions   (OpenAIChatResponder)
    a token count     <- POST {base}/tokenize              (ServerTokeniser)

Both are written against the standard library's ``urllib``, because ``p1-verifiers/src`` is
inside the import rule (``scripts/check_repo.py --imports``): a project may import a
framework, this layer may not. The serving stack already speaks HTTP, so nothing is lost -
what would be lost is the ability to run the checks on a machine that has no framework
installed, which is the property that lets you tell an environment problem from a bug.

Four rules this module keeps:

* **A key is read from an environment variable, never passed in, never defaulted, and never
  printed.** A local server usually needs none, and *which* of the two it was is recorded
  rather than assumed.
* **An empty answer is an error, not an answer.** Returning ``""`` would be scored as a wrong
  attempt and quietly pollute the reward; the failure belongs at the source.
* **A retry is for the server's problems, never for yours.** 5xx and network errors retry
  with backoff; 4xx does not, because the request is wrong and repeating it wastes the GPU.
* **A token count comes with the name of what counted it.** ``ServerTokeniser.name`` goes
  into ``env_version``, so a row can never claim a count whose author is unknown.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .harness import SYSTEM_PROMPT, Task


class RemoteError(RuntimeError):
    """The server could not be used, with a message that says which part failed.

    ``status`` is the HTTP status when there was one, and ``None`` for a network failure -
    the difference decides whether a retry makes sense.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def post_json(url: str, payload: dict, *, headers: dict[str, str] | None = None,
              timeout: float = 30.0) -> dict:
    """POST JSON and return the decoded answer, or raise ``RemoteError`` saying what happened.

    One function for both clients, so retry policy, error text and body truncation are
    defined once instead of twice.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read()[:200].decode("utf-8", "replace").strip()
        finally:
            exc.close()  # otherwise the socket waits for the garbage collector, loudly
        raise RemoteError(f"{url} answered HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise RemoteError(f"{url} is unreachable: {exc.reason}") from exc
    except TimeoutError as exc:  # a bare socket timeout, not wrapped by urllib
        raise RemoteError(f"{url} timed out after {timeout:g}s") from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"{url} did not answer with JSON: {raw[:120]!r}") from exc
    if not isinstance(decoded, dict):
        raise RemoteError(f"{url} answered with {type(decoded).__name__}, not an object")
    return decoded


def get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0) -> dict:
    """GET and decode JSON, sharing ``post_json``'s error text and body truncation."""
    request = urllib.request.Request(url, method="GET", headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read()[:200].decode("utf-8", "replace").strip()
        finally:
            exc.close()
        raise RemoteError(f"{url} answered HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise RemoteError(f"{url} is unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RemoteError(f"{url} timed out after {timeout:g}s") from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"{url} did not answer with JSON: {raw[:120]!r}") from exc
    if not isinstance(decoded, dict):
        raise RemoteError(f"{url} answered with {type(decoded).__name__}, not an object")
    return decoded


def list_models(base_url: str, *, timeout: float = 15.0) -> list[str]:
    """The model ids a server is serving.

    Used as a preflight, because the failure it prevents is the most annoying one in this
    whole area: a run that spends twenty minutes fetching weights before discovering that the
    name in the config is not the name the server answers to.
    """
    data = get_json(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    entries = data.get("data")
    if not isinstance(entries, list):
        raise RemoteError(f"{base_url}/v1/models answered without a 'data' list: {sorted(data)}")
    return [str(entry.get("id")) for entry in entries if isinstance(entry, dict) and entry.get("id")]


class OpenAIChatResponder:
    """A responder backed by any OpenAI-compatible chat endpoint.

    Works against vLLM, SGLang, llama.cpp's server, or anything else that answers
    ``/v1/chat/completions``; the point is that P1 does not care which.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        timeout: float = 120.0,
        retries: int = 2,
        seed: int | None = None,
        sleep=time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.seed = seed
        self._sleep = sleep
        #: Whatever the last call reported. Provenance for a run record, not a decision input.
        self.last_usage: dict | None = None
        self.last_finish_reason: str | None = None

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def describe(self) -> str:
        """One line for a run record: what answered, and whether a key was sent.

        Deliberately reports the *name* of the environment variable and whether it was set
        - never the value, and never a hash of it either, which would be a liveness oracle.
        """
        if not self.api_key_env:
            key = "no key (no env var configured)"
        elif os.environ.get(self.api_key_env):
            key = f"key from ${self.api_key_env}"
        else:
            key = f"no key (${self.api_key_env} is unset)"
        return (f"{self.model} at {self.url}, temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, {key}")

    def _headers(self) -> dict[str, str]:
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return {"Authorization": f"Bearer {value}"}
        return {}

    def _retryable(self, error: RemoteError) -> bool:
        return error.status is None or error.status >= 500

    def respond(self, task: Task, sample_index: int) -> str:
        """One attempt at one task. Raises ``RemoteError`` rather than inventing an answer."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task.prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed + sample_index

        attempt = 0
        while True:
            try:
                data = post_json(self.url, payload, headers=self._headers(), timeout=self.timeout)
                break
            except RemoteError as exc:
                if attempt >= self.retries or not self._retryable(exc):
                    raise
                attempt += 1
                self._sleep(min(2.0 ** attempt, 8.0))

        self.last_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RemoteError(f"{self.url} answered without choices: {sorted(data)}")
        first = choices[0] if isinstance(choices[0], dict) else {}
        self.last_finish_reason = first.get("finish_reason")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RemoteError(
                f"{self.url} answered with no text (finish_reason={self.last_finish_reason!r}); "
                f"an empty attempt would be scored as a wrong answer, so it is an error here"
            )
        return content


class ServerTokeniser:
    """Token counts from the server's own tokeniser, over HTTP.

    Using the endpoint rather than a local tokeniser library keeps this layer dependency-free
    and guarantees the count comes from the same model that produced the text - which is the
    whole reason the fake counter was unacceptable for a real run.

    ``/tokenize`` is what vLLM exposes; ``/v1/tokenize`` is tried as well because a proxy can
    mount the OpenAI routes under a prefix. The endpoint that answered is remembered.
    """

    def __init__(self, base_url: str, model: str, *, timeout: float = 30.0,
                 endpoints: tuple[str, ...] = ("/tokenize", "/v1/tokenize")) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoints = endpoints
        self._found: str | None = None
        # Host, not netloc: the port is incidental (restart the server and it changes), and a
        # tokeniser's identity should not move because a service was rebound. The model and the
        # machine are what a row needs to be traceable.
        host = urlparse(self.base_url).hostname or self.base_url
        #: Goes into ``env_version``: a token count is only meaningful next to what counted it.
        self.name = f"server-tokenize/{model}@{host}"

    def count(self, text: str) -> int:
        """How many tokens this server would make of ``text``."""
        if self._found:
            return self._count_at(self._found, text)

        problems: list[str] = []
        for endpoint in self._endpoints:
            try:
                count = self._count_at(endpoint, text)
            except RemoteError as exc:
                problems.append(f"{endpoint}: {exc}")
                continue
            self._found = endpoint
            return count
        raise RemoteError(
            "no working tokenize endpoint. Tried: " + " | ".join(problems) +
            ". A server without one cannot supply the token counts a mask needs; either "
            "enable it or accept a run that says pseudo-words/v1 in env_version."
        )

    def _count_at(self, endpoint: str, text: str) -> int:
        data = post_json(f"{self.base_url}{endpoint}", {"model": self.model, "prompt": text},
                         timeout=self.timeout)
        if isinstance(data.get("count"), int):
            return data["count"]
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise RemoteError(f"{self.base_url}{endpoint} answered without a count or tokens: "
                          f"{sorted(data)}")
