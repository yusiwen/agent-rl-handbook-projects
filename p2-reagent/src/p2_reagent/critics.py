"""Critics: the free signal, the scripted stand-in, the served model, and the budget.

Chapter 6, Lesson 3: the reward model's three outputs are a **reasoning trace**, a **critique** and
a **score** - and the point of asking for three is that a single number cannot be debugged.  This
module makes that shape the interface, so a rule critic, a fixture and a served model are
interchangeable behind it and the training loop never learns which one it is consuming.

Chapter 6, Lesson 3 also asks for the budget: score everything with the cheap critic, then spend
the expensive one only where it can change the outcome - the uncertain middle.  ``BudgetedCritic``
is that policy, and it writes its decision into the verdict so that "the gate became a learned
surface" is something a log can show rather than something a reviewer has to infer.

The model critic never invents a score.  A malformed reply raises ``CriticParseError``: a judge
that returns 0.0 on a parse failure turns a bug into a training signal, which is the one failure
mode in this whole project that a reward curve cannot reveal.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Sequence

from .agent import Episode

#: A turn's score is centred here and moved by how much closer to the goal it left the world.
#: Centring at 0.5 rather than 0 matters: a turn that changes nothing is not a failure, it is a
#: turn that taught nothing, and a critic that scores it zero teaches a policy to avoid reading
#: before acting.  The measured reason this is a *delta* and not "the state changed": with an
#: activity-based score, the demo policy learned to repeat a harmless action until the turn cap
#: stopped it - Chapter 6, Lesson 4's Echo Trap, reproduced accidentally in the first draft.
TURN_CENTRE = 0.5


class CriticParseError(RuntimeError):
    """The critic's reply could not be read.  Never downgraded to a score."""


@dataclass(frozen=True)
class TurnNote:
    """One turn's critique: the index, the score, and the sentence a human can act on."""

    index: int
    score: float
    critique: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"turn {self.index}: score {self.score} is outside [0, 1]")


@dataclass(frozen=True)
class Verdict:
    """A reward model's answer for one attempt: trace, critique, score, and its provenance."""

    score: float
    critique: str
    trace: str
    per_turn: list[TurnNote] = field(default_factory=list)
    source: str = "unknown"
    #: Why the expensive critic did or did not run.  Empty for critics that are always cheap.
    gate: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score {self.score} is outside [0, 1]")

    def turn_scores(self) -> list[float]:
        return [note.score for note in self.per_turn]

    def render(self) -> str:
        head = f"[{self.source}] score={self.score:.2f}"
        if self.gate:
            head += f"  ({self.gate})"
        lines = [head, f"  why: {self.critique}"]
        for note in self.per_turn:
            lines.append(f"  turn {note.index}: {note.score:.2f}  {note.critique}")
        return "\n".join(lines)


class RuleCritic:
    """No model at all: reads the environment's signals.  The cheapest dense signal there is.

    This is Chapter 6, Lesson 2's "free one hiding in the table".  It is also the honest baseline
    for the served critic: if a model judge cannot beat this, its extra cost bought nothing.
    """

    source = "rule"

    def critique(self, episode: Episode) -> Verdict:
        notes: list[TurnNote] = []
        trace = episode.attempt_progress
        for index, signal in enumerate(episode.signals):
            if signal == "error":
                score = 0.0
            elif signal == "done":
                # The final turn carries the outcome: a `done` that reported the wrong answer is not
                # a success, and saying so here keeps the rule critic from congratulating a run that
                # failed while looking tidy.
                score = 1.0 if episode.solved else 0.0
            else:
                delta = (trace[index + 1] - trace[index]) if index + 1 < len(trace) else 0.0
                score = min(1.0, max(0.0, TURN_CENTRE + delta))
            notes.append(TurnNote(index=index, score=score, critique=_signal_sentence(signal, index, score)))
        score = sum(note.score for note in notes) / len(notes) if notes else 0.0
        return Verdict(
            score=score,
            critique=(
                f"{len(notes)} turn(s) judged from the environment's own signals; "
                f"outcome {'met' if episode.solved else 'not met'}"
            ),
            trace="signals: " + ", ".join(f"t{i}={signal}" for i, signal in enumerate(episode.signals)),
            per_turn=notes,
            source=self.source,
        )


def _signal_sentence(signal: str, index: int, score: float) -> str:
    if signal == "error":
        return f"turn {index}: the environment rejected the action"
    if signal == "done":
        return f"turn {index}: the episode was closed by a report (score {score:.2f})"
    if score > TURN_CENTRE:
        return f"turn {index}: the action moved the world towards the goal ({score - TURN_CENTRE:+.2f})"
    if score < TURN_CENTRE:
        return f"turn {index}: the action moved the world away from the goal ({score - TURN_CENTRE:+.2f})"
    return f"turn {index}: the action ran and changed nothing that matters"


class FixtureCritic:
    """A scripted stand-in for a model judge, so the whole pipeline runs offline.

    It reads a JSONL fixture keyed by ``(task_id, turn_index)`` and falls back to a rubric when a
    pair is missing.  Two reasons it is not a mock in the dismissive sense: it produces *text* like
    a judge does (the interface is exercised, not bypassed), and its rubric is deliberately not the
    rule critic's - it rewards turns that *look* productive, which is how the conflicts Chapter 6
    asks the reader to find actually appear in the log.
    """

    source = "fixture"

    def __init__(self, scripted: dict[tuple[str, int], tuple[float, str]] | None = None) -> None:
        self.scripted = scripted or {}

    @classmethod
    def from_jsonl(cls, path: str) -> "FixtureCritic":
        scripted: dict[tuple[str, int], tuple[float, str]] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scripted[(str(row["task"]), int(row["turn"]))] = (
                    float(row["score"]),
                    str(row["critique"]),
                )
        return cls(scripted)

    def critique(self, episode: Episode) -> Verdict:
        notes: list[TurnNote] = []
        for index, signal in enumerate(episode.signals):
            key = (episode.task.task_id, index)
            if key in self.scripted:
                score, text = self.scripted[key]
            else:
                # The rubric: a turn whose action produced a visible change reads as competent,
                # whether or not it moved the task forward.  That is a real judge's failure mode.
                changed = signal in {"ok", "done"}
                score = 0.9 if changed else 0.25
                text = (
                    f"turn {index}: the action produced a visible change, which reads as progress"
                    if changed
                    else f"turn {index}: the action produced no visible change"
                )
            notes.append(TurnNote(index=index, score=score, critique=text))
        score = sum(note.score for note in notes) / len(notes) if notes else 0.0
        return Verdict(
            score=score,
            critique=f"{len(notes)} turn(s) judged from the scripted critique set",
            trace="scripted judge; no model",
            per_turn=notes,
            source=self.source,
        )


class ModelCritic:
    """The real thing: an OpenAI-compatible endpoint asked for a score and a reason per turn.

    The reply format is demanded and parsed strictly::

        TURN <n> SCORE: <0..1>
        WHY: <one sentence>

    Anything else raises.  The endpoint's key comes from the environment and is never logged.

    ``api_key_env=None`` is the local case, and it is not the same as an empty key: a vLLM server
    on the same box authenticates nobody, and refusing to call it because a variable is unset would
    make the judge unusable exactly where it is cheapest to run. Naming *which* mode is in force is
    the point - a missing key where one was expected still raises.
    """

    source = "model"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})

    # ------------------------------------------------------------------ the request
    def _prompt(self, episode: Episode) -> str:
        lines = [
            "You are judging a multi-turn agent attempt. For each turn, say whether it moved the",
            "task forward. Reply with exactly one block per turn:",
            "TURN <n> SCORE: <a number between 0 and 1>",
            "WHY: <one sentence>",
            "",
            f"TASK: {episode.task.prompt}",
            f"GOAL: {episode.task.goal}",
            "",
        ]
        for index, signal in enumerate(episode.signals):
            action = episode.actions[index]
            observation = episode.turns[index]["observation"]
            lines.append(f"TURN {index}: action={action!r} observation={observation!r} signal={signal}")
        return "\n".join(lines)

    def _call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise CriticParseError(
                    f"no API key in ${self.api_key_env}; refusing to call the judge without one "
                    "(and never printing a key that does exist)"
                )
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_body,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise CriticParseError(f"the judge answered HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise CriticParseError(f"could not reach the judge at {self.base_url}: {exc.reason}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CriticParseError(f"the judge's reply has no message content: {str(payload)[:200]}") from exc

    # ------------------------------------------------------------------ the reply
    _TURN = re.compile(r"TURN\s+(\d+)\s+SCORE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    _WHY = re.compile(r"WHY:\s*(.+)", re.IGNORECASE)

    @classmethod
    def parse(cls, reply: str, expected_turns: int) -> list[TurnNote]:
        scores = {int(match.group(1)): float(match.group(2)) for match in cls._TURN.finditer(reply)}
        if sorted(scores) != list(range(expected_turns)):
            raise CriticParseError(
                f"the judge scored turn(s) {sorted(scores)} but the attempt has {expected_turns}; "
                "a partial judgement is not a judgement"
            )
        whys = [match.group(1).strip() for match in cls._WHY.finditer(reply)]
        notes: list[TurnNote] = []
        for index in range(expected_turns):
            text = whys[index] if index < len(whys) else "(no reason given)"
            notes.append(TurnNote(index=index, score=scores[index], critique=text))
        return notes

    def critique(self, episode: Episode) -> Verdict:
        reply = self._call(self._prompt(episode))
        notes = self.parse(reply, len(episode.actions))
        score = sum(note.score for note in notes) / len(notes) if notes else 0.0
        return Verdict(
            score=score,
            critique=f"{len(notes)} turn(s) judged by {self.model}",
            trace=reply.strip(),
            per_turn=notes,
            source=self.source,
        )


class BudgetedCritic:
    """Cheap first, expensive only in the uncertain middle (Chapter 6, Lesson 3).

    The gate is a *policy decision* and it is logged: every verdict says whether the expensive
    critic ran and on which rule score it decided.  When the expensive critic does not run, the
    cheap verdict is returned unchanged - not blended, because a blended verdict would make the
    two sources indistinguishable in the log, which is the one thing the chapter insists on.
    """

    def __init__(
        self,
        cheap: RuleCritic,
        expensive: "FixtureCritic | ModelCritic",
        *,
        low: float = 0.5,
        high: float = 0.9,
    ) -> None:
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("the budget needs 0 <= low < high <= 1")
        self.cheap = cheap
        self.expensive = expensive
        self.low = low
        self.high = high

    def critique(self, episode: Episode) -> Verdict:
        cheap = self.cheap.critique(episode)
        uncertain = self.low <= cheap.score <= self.high
        if not uncertain:
            return Verdict(
                score=cheap.score,
                critique=cheap.critique,
                trace=cheap.trace,
                per_turn=cheap.per_turn,
                source=cheap.source,
                gate=f"expensive critic skipped: rule score {cheap.score:.2f} is outside [{self.low}, {self.high}]",
            )
        expensive = self.expensive.critique(episode)
        return Verdict(
            score=expensive.score,
            critique=expensive.critique,
            trace=expensive.trace,
            per_turn=expensive.per_turn,
            source=expensive.source,
            gate=f"expensive critic ran: rule score {cheap.score:.2f} is inside [{self.low}, {self.high}]",
        )


def disagreement(rule: Verdict, other: Verdict, *, margin: float = 0.25) -> list[str]:
    """Turn-level disagreements big enough to be worth reading.

    This is Chapter 5's conflict detection one level down, and Chapter 6's pass criterion asks for
    exactly this: one example of a rule-vs-judge conflict, found and acted on rather than averaged
    away.  A conflict here is the same turn scored at least ``margin`` apart.
    """
    conflicts: list[str] = []
    for left, right in zip(rule.per_turn, other.per_turn):
        if abs(left.score - right.score) >= margin:
            conflicts.append(
                f"turn {left.index}: {rule.source}={left.score:.2f} vs {other.source}={right.score:.2f}"
                f"  ({left.critique} / {right.critique})"
            )
    return conflicts
