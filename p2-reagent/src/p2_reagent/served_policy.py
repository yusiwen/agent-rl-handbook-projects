"""A policy that is an HTTP endpoint.

The loop, the credit machinery and the checks were built against a table of action probabilities.
This module swaps the table for the fine-tuned checkpoint without touching any of them: it answers
the same `choose` call, one request per turn, and hands back exactly what the model said.

Two consequences, both stated rather than papered over:

* **There is no update.** `apply` does nothing, because the policy is a remote endpoint and the
  weight the loop computes has nowhere to go. So `Config.served` switches off the two checks that
  assert a *learning* loop - lineage ("the update changed the parameters") and entropy ("one round
  did not halve it") - and this class reports a distinct-action ratio in their place. A frozen
  policy has no entropy to measure; it has a repertoire, and the ratio says how wide it is.
* **An unparsed reply is still an action.** It is passed through as the text the model produced, so
  the environment rejects it and the trace records it. Repaired replies would make a policy that
  cannot hold the format look better than it is.

Every call is counted by reason, because the interesting failure in a served loop is not the wrong
action, it is the action that was never parseable in the first place.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter

from . import actions, prompt


class ServedPolicy:
    """Chooses actions by asking an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 64,
        timeout: int = 180,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.version = f"served:{model}"
        self.calls = 0
        self.reasons: Counter[str] = Counter()
        self.taken: Counter[str] = Counter()
        self.seconds = 0.0

    # ------------------------------------------------------------------ the loop's interface
    def choose(self, attempt) -> str:
        history = [(action, result.signal, result.observation) for action, result in attempt.history]
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu(), history)
        reply = self._ask(messages)
        parsed = actions.parse_action(reply, attempt.menu())
        self.reasons[parsed.reason] += 1
        action = parsed.action if parsed.ok else (parsed.raw or "<unparsed>")
        self.taken[action] += 1
        return action

    def sample(self, signature: str, menu) -> str:
        raise RuntimeError(
            "ServedPolicy needs the attempt, not a state signature; run_episode passes one when "
            "the policy defines `choose`"
        )

    def apply(self, updates) -> None:
        """A served policy is not trained here. The update is computed and goes nowhere."""
        return None

    # ------------------------------------------------------------------ what it measured
    @property
    def distinct_action_ratio(self) -> float:
        """Distinct actions divided by turns: the repertoire, where a table would report entropy."""
        total = sum(self.taken.values())
        return len(self.taken) / total if total else 0.0

    def summary(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "seconds": round(self.seconds, 1),
            "distinct_actions": len(self.taken),
            "distinct_action_ratio": round(self.distinct_action_ratio, 4),
            "reasons": dict(self.reasons),
        }

    # ------------------------------------------------------------------ the wire
    def _ask(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                self.calls += 1
                self.seconds += time.time() - started
                return parsed["choices"][0]["message"].get("content") or ""
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        raise RuntimeError(f"served policy could not reach {self.base_url}: {last}")
