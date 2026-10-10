"""Run records: what makes a number in this repo mean something later.

Chapter 10's fourth milestone asks for a run a third party can reproduce - or refute.
Reproduce *what*?  Not the loss curve: the setup.  So every run writes a header (what
was run, on what, at which code revision, over which data) and one line per round
(what came out).

Two details matter more than they look:

* **Sample ids are stable across rounds.**  A round record lists the sample ids it
  trained on.  Comparing two rounds by ids is how you check that the feedback you
  added actually reached the next batch instead of being consumed by a logging bug.
* **The spread is reported per round.**  A GRPO group whose rewards are all equal
  produces no gradient at all - the advantage is zero by construction (Chapter 4).
  Reporting the fraction of groups with a non-zero spread catches the most common
  silent stall: training runs, loss moves, nothing is learned.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import Rollout

#: Spreads below this are treated as zero.  Reward terms are floats; exact equality
#: is a coincidence, not a measurement.
SPREAD_EPSILON = 1e-9


def utc_now() -> str:
    """Timestamps are UTC and explicit, because run records travel between machines."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def digest_of(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """A short content digest, for pinning a dataset without committing it."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()[:16]


@dataclass(frozen=True)
class RunHeader:
    """Everything needed to rerun this, written once per run.

    ``env_version``, ``config_hash`` and ``code_sha`` are required with no defaults:
    a run record without them is a number with no owner.
    """

    run_id: str
    env_version: str
    config_hash: str
    code_sha: str
    project: str
    started_at: str = field(default_factory=utc_now)
    dataset: str = "fixtures"
    dataset_digest: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundRecord:
    """One training round's outcome, in terms a human can act on."""

    round_index: int
    reward_mean: float
    term_means: dict[str, float]
    spread_fraction: float
    groups: int
    rollouts: int
    tokens: int
    trained_tokens: int
    sample_ids: list[str]
    recorded_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        terms = ", ".join(f"{name}={value:+.3f}" for name, value in sorted(self.term_means.items()))
        return (
            f"round {self.round_index}: reward={self.reward_mean:+.3f} ({terms})  "
            f"spread={self.spread_fraction:.0%} of {self.groups} groups  "
            f"{self.trained_tokens}/{self.tokens} tokens trained"
        )


def sample_id(rollout: Rollout) -> str:
    """A stable identifier for one sampled attempt: prompt plus sample index."""
    return f"{rollout.prompt_id}#{rollout.sample_index}"


def sample_ids(rollouts: Iterable[Rollout]) -> list[str]:
    return sorted(sample_id(rollout) for rollout in rollouts)


def spread_per_group(rollouts: Iterable[Rollout]) -> dict[str, float]:
    """Per GRPO group: the gap between the best and worst attempt.

    This is the quantity the advantage is built from.  Zero means the group cannot
    teach anything.
    """
    grouped: dict[str, list[float]] = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.group_id, []).append(rollout.reward())
    return {
        group_id: (max(values) - min(values)) if values else 0.0
        for group_id, values in grouped.items()
    }


def spread_fraction(rollouts: Iterable[Rollout]) -> float:
    """Share of groups that can produce a gradient at all."""
    spreads = spread_per_group(rollouts)
    if not spreads:
        return 0.0
    usable = sum(1 for value in spreads.values() if value > SPREAD_EPSILON)
    return usable / len(spreads)


def round_record(round_index: int, rollouts: Sequence[Rollout]) -> RoundRecord:
    """Summarise one round.  Means are over rollouts, not over tokens."""
    if not rollouts:
        raise ValueError("cannot summarise an empty round")
    totals = {rollout.episode_id: rollout for rollout in rollouts}
    if len(totals) != len(rollouts):
        raise ValueError("duplicate episode_id in one round; ids must identify an attempt")

    term_means: dict[str, float] = {}
    for rollout in rollouts:
        for name, value in rollout.reward_terms.items():
            term_means[name] = term_means.get(name, 0.0) + value
    for name in term_means:
        term_means[name] /= len(rollouts)

    rewards = [rollout.reward() for rollout in rollouts]
    return RoundRecord(
        round_index=round_index,
        reward_mean=sum(rewards) / len(rewards),
        term_means=term_means,
        spread_fraction=spread_fraction(rollouts),
        groups=len({rollout.group_id for rollout in rollouts}),
        rollouts=len(rollouts),
        tokens=sum(rollout.total_tokens for rollout in rollouts),
        trained_tokens=sum(rollout.model_tokens for rollout in rollouts),
        sample_ids=sample_ids(rollouts),
    )


class RunWriter:
    """Writes ``<root>/<run_id>/{header.json,rounds.jsonl}``.

    ``root`` defaults to ``runs/``, which ``.gitignore`` keeps out of the repository:
    the *format* is versioned, the *rows* are not.
    """

    def __init__(self, header: RunHeader, root: str | os.PathLike[str] = "runs") -> None:
        self.header = header
        self.dir = Path(root) / header.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "header.json").write_text(
            json.dumps(header.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._rounds_path = self.dir / "rounds.jsonl"

    def append_round(self, record: RoundRecord) -> None:
        with open(self._rounds_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")

    def append_rollouts(self, round_index: int, rollouts: Sequence[Rollout]) -> None:
        """Persist the raw rows of a round, so a claim can be re-derived later."""
        path = self.dir / f"round-{round_index:03d}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(rollout.to_json())
                handle.write("\n")

    def read_rounds(self) -> list[dict[str, Any]]:
        if not self._rounds_path.exists():
            return []
        return [
            json.loads(line)
            for line in self._rounds_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
