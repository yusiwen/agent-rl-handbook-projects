"""Turning a task set into rollout rows: the loop P1 owns.

The pipeline is deliberately dull, because everything interesting in the book is
somewhere else:

    task -> responder -> verifier -> reward terms -> rollout row -> run record

Four parts are pluggable: the **responder** (the model; a fixture responder in the tests so
nothing needs a GPU or a network), the **registry** (the verifier suite), the **task set** (a
JSONL file), and the **tokeniser** (the server's own in a real run, a word counter in the
tests).  ``remote.py`` holds the two HTTP-backed ones.

What this module refuses to do:

* It will not invent a reward.  A task whose verifier declines it is reported as
  uncovered, not scored zero (see ``common.verifier.CoverageReport``).
* It will not write a row whose loss mask is wrong.  The mask is built from the
  segments and then *checked* against them before the row leaves the function; a
  pipeline that records its own bug is worth more than one that hides it.
* It will not pretend to have a tokeniser.  ``PseudoTokeniser`` counts words and says so in
  ``env_version``; a real run tokenises with the model's own tokeniser and records its
  version, because a token count is only meaningful next to the thing that counted.  Supply
  a real tokeniser and leave the marker alone and the run stops rather than mislabelling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from common.mask import build_mask, mask_faults, turn_boundaries
from common.schema import Rollout, Segment
from common.verifier import MissingVerifier, VerifierRegistry

#: Marks the stand-in token counter in every row it produces.  Never reuse this value
#: for a run that used a real tokeniser.
PSEUDO_TOKENISER = "pseudo-words/v1 (no tokeniser)"

SYSTEM_PROMPT = "Answer the question. Put the final answer on its own line."


class TaskSetError(ValueError):
    """A task file that cannot be used as written."""


@dataclass(frozen=True)
class Task:
    """One row of a task file.

    ``fields`` keeps the rest of the row (the answer, a tolerance, a test command)
    without this class having to know every verifier's vocabulary.
    """

    id: str
    prompt: str
    verifier: str
    fields: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {"id": self.id, "prompt": self.prompt, "verifier": self.verifier, **self.fields}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        missing = [key for key in ("id", "prompt", "verifier") if not str(data.get(key, "")).strip()]
        if missing:
            raise TaskSetError(f"task is missing {', '.join(missing)}: {dict(data)!r}")
        known = {"id", "prompt", "verifier"}
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            verifier=str(data["verifier"]),
            fields={key: value for key, value in data.items() if key not in known},
        )


def load_tasks(path: str | Path) -> list[Task]:
    """Read a JSONL task file.  A malformed row names its line number and stops the run."""
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            task = Task.from_dict(json.loads(line))
        except (json.JSONDecodeError, TaskSetError) as exc:
            raise TaskSetError(f"{path}:{lineno}: {exc}") from exc
        if task.id in seen:
            raise TaskSetError(f"{path}:{lineno}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    return tasks


class Responder(Protocol):
    """Whatever produces an attempt.  A model in a real run, a fixture in the tests."""

    def respond(self, task: Task, sample_index: int) -> str:  # pragma: no cover - protocol
        ...


class FixtureResponder:
    """Answers from a ``{prompt_id: [response, ...]}`` mapping, cycling by sample index.

    Canned answers keep the checks runnable with no network, no weights and no GPU -
    which is what lets you tell an environment problem from an implementation problem
    before you spend an hour on a model download.
    """

    def __init__(self, answers: Mapping[str, Sequence[str]]) -> None:
        if not answers:
            raise ValueError("a fixture responder needs at least one answer")
        self._answers = {key: tuple(values) for key, values in answers.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "FixtureResponder":
        grouped: dict[str, list[str]] = {}
        for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                grouped.setdefault(str(row["prompt_id"]), []).append(str(row["response"]))
            except KeyError as exc:
                raise TaskSetError(f"{path}:{lineno}: response row is missing {exc}") from exc
        return cls(grouped)

    def respond(self, task: Task, sample_index: int) -> str:
        answers = self._answers.get(task.id)
        if not answers:
            raise TaskSetError(f"the fixture responder has no answer for task {task.id!r}")
        return answers[sample_index % len(answers)]


class Tokeniser(Protocol):
    """Whatever counts tokens.

    ``name`` is not decoration: it is what goes into a row's ``env_version``, so a count can
    always be traced to the thing that produced it.
    """

    name: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


def _pseudo_tokens(text: str) -> int:
    """A stand-in token count: words, floored at one.

    Not a tokeniser.  Every row it feeds carries ``PSEUDO_TOKENISER`` in ``env_version``
    so nobody mistakes these counts for a measurement.
    """
    return max(1, len(text.split()))


class PseudoTokeniser:
    """The stand-in, in the shape the pipeline now expects.

    Kept as the default so that every offline check keeps working unchanged, and so that
    "no tokeniser here" is a state the code can name rather than a state it forgets.
    """

    name = PSEUDO_TOKENISER

    def count(self, text: str) -> int:
        return _pseudo_tokens(text)


#: The default tokeniser.  Compared by identity in ``run_episode`` to catch a real tokeniser
#: whose rows would still carry the stand-in's marker.
PSEUDO = PseudoTokeniser()


def segments_for(task: Task, response: str, tokeniser: Tokeniser = PSEUDO) -> list[Segment]:
    """A single-turn trace: the instruction, the question, the attempt.

    Single-turn because P1's tasks are single-turn.  Multi-turn traces (tool calls,
    environment replies) are P2's and P5's shape; the record is the same either way.
    """
    return [
        Segment(role="system", text=SYSTEM_PROMPT, tokens=tokeniser.count(SYSTEM_PROMPT)),
        Segment(role="user", text=task.prompt, tokens=tokeniser.count(task.prompt)),
        Segment(role="model", text=response, tokens=tokeniser.count(response)),
    ]


@dataclass(frozen=True)
class Provenance:
    """What produced a run.  Required, with no defaults, for the same reason as in runs.py."""

    env_version: str = PSEUDO_TOKENISER
    config_hash: str = "unset"
    code_sha: str = "unset"


def run_episode(
    task: Task,
    response: str,
    registry: VerifierRegistry,
    sample_index: int,
    provenance: Provenance,
    tokeniser: Tokeniser | None = None,
) -> Rollout:
    """Score one attempt and build its row.

    Raises ``MissingVerifier`` when the task's verifier is not registered: a task that
    cannot be scored is a gap in the suite, not a zero.
    """
    tokeniser = tokeniser or PSEUDO
    if tokeniser is not PSEUDO and provenance.env_version == PSEUDO_TOKENISER:
        raise ValueError(
            "a tokeniser was supplied but the row's env_version still says "
            f"{PSEUDO_TOKENISER!r}. Pass Provenance(env_version=tokeniser.name, ...) so the "
            "row names what counted its tokens; a count whose author is unknown is not a "
            "measurement, it is a number."
        )

    verifier = registry.for_task(task.as_mapping())
    if not verifier.can_verify(task.as_mapping()):
        raise MissingVerifier(
            f"{verifier.name!r} cannot verify task {task.id!r} as written; either the "
            "task is missing a field the verifier needs, or the verifier is still a stub"
        )
    outcome = verifier.verify(task.as_mapping(), response)

    segments = segments_for(task, response, tokeniser)
    mask = build_mask(segments)
    faults = mask_faults(segments, mask)
    if faults:  # a bug here means the trace and the mask disagree about who wrote what
        raise AssertionError(f"mask built from the segments does not match them: {faults[:3]}")

    return Rollout(
        episode_id=f"{task.id}-s{sample_index}",
        prompt_id=task.id,
        sample_index=sample_index,
        group_id=task.id,
        segments=segments,
        reward_terms=outcome.as_terms("rule"),
        env_version=provenance.env_version,
        config_hash=provenance.config_hash,
        code_sha=provenance.code_sha,
        mask=mask,
        turn_count=len(turn_boundaries(segments)),
        extras={
            "verifier": verifier.name,
            "passed": outcome.passed,
            "detail": outcome.detail,
            "response": response,
        },
    )


def run_taskset(
    tasks: Iterable[Task],
    responder: Responder,
    registry: VerifierRegistry,
    samples_per_task: int = 1,
    provenance: Provenance | None = None,
    tokeniser: Tokeniser | None = None,
) -> list[Rollout]:
    """Run every task ``samples_per_task`` times and return the rows.

    With more than one sample per task the rows form GRPO groups: same ``group_id``,
    different ``sample_index``.  That is what makes an advantage computable at all.
    """
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be at least 1")
    provenance = provenance or Provenance()
    rows: list[Rollout] = []
    for task in tasks:
        for sample_index in range(samples_per_task):
            rows.append(
                run_episode(task, responder.respond(task, sample_index), registry,
                            sample_index, provenance, tokeniser)
            )
    return rows


def partition_by_coverage(
    tasks: Iterable[Task], registry: VerifierRegistry
) -> tuple[list[Task], list[tuple[Task, str]]]:
    """Split a task set into what this suite can score and what it cannot, with reasons.

    A real pipeline does this before training, not after: the covered subset is what the
    run can learn from, and the uncovered remainder is the honest answer to "what is my
    ceiling".  Returning the reasons matters - "declared a verifier that declined it" and
    "names no verifier at all" are different kinds of gap.
    """
    covered: list[Task] = []
    uncovered: list[tuple[Task, str]] = []
    for task in tasks:
        mapping = task.as_mapping()
        try:
            verifier = registry.for_task(mapping)
        except MissingVerifier as exc:
            uncovered.append((task, str(exc)))
            continue
        if verifier.can_verify(mapping):
            covered.append(task)
        else:
            uncovered.append((task, f"{verifier.name!r} cannot verify this task as written"))
    return covered, uncovered


def pass_rate(rollouts: Sequence[Rollout]) -> tuple[int, int]:
    """``(passed, total)`` counted from the recorded outcomes, not from the reward."""
    passed = sum(1 for row in rollouts if row.extras.get("passed") is True)
    return passed, len(rollouts)


def report(rollouts: Sequence[Rollout]) -> str:
    """A one-screen summary of a run, for the check script and for your own logs."""
    if not rollouts:
        return "no rollouts"
    passed, total = pass_rate(rollouts)
    rewards = [row.reward() for row in rollouts]
    failures = [row for row in rollouts if row.extras.get("passed") is not True]
    lines = [
        f"rollouts: {total}  passed: {passed}/{total} ({passed / total:.0%})",
        f"reward: min={min(rewards):+.2f} max={max(rewards):+.2f} "
        f"mean={sum(rewards) / len(rewards):+.2f}",
        f"tokens: {sum(row.model_tokens for row in rollouts)} trained of "
        f"{sum(row.total_tokens for row in rollouts)} total",
    ]
    for row in failures[:5]:
        lines.append(f"  missed: {row.episode_id} - {row.extras.get('detail', '')}")
    if len(failures) > 5:
        lines.append(f"  ... and {len(failures) - 5} more")
    return "\n".join(lines)
