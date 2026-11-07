"""A chat client for whatever OpenAI-compatible server is serving the policy.

Small on purpose: one function, one retry policy, and a ``ScriptedClient`` so the agent loop
can be tested without a GPU, a server or a network. ``chat_template_kwargs`` turns Qwen's
thinking mode off; a reasoning model that spends its turn thinking does not emit the markers,
which shows up as a protocol error rather than as a wrong answer - see Chapter 10, Lesson 3.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        retries: int = 2,
        disable_thinking: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.disable_thinking = disable_thinking
        self.calls = 0

    def ask(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    parsed = json.loads(response.read().decode("utf-8"))
                self.calls += 1
                return (parsed["choices"][0]["message"].get("content") or "").strip()
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        raise RuntimeError(f"could not reach {self.base_url}: {last}")


class ScriptedClient:
    """Returns prepared replies in order.  Used by the tests, and by nothing else."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list[dict[str, str]]] = []

    def ask(self, messages: list[dict[str, str]]) -> str:
        self.seen.append(list(messages))
        if self.calls >= len(self.replies):
            raise AssertionError(f"scripted client ran out of replies after {self.calls}")
        reply = self.replies[self.calls]
        self.calls += 1
        return reply
