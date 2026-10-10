"""The one row every project writes: a rollout.

Six projects, one record shape.  If each project invented its own row, every
diagnostic in the book (advantage spread, per-term reward, mask integrity, turn
boundaries) would be written six times and would drift apart.  This is that row.

Design decisions worth knowing:

* **Rewards are terms, never a single number.**  ``reward_terms`` is a mapping such
  as ``{"rule": 1.0, "judge": -0.2}``.  The total is *derived* (see ``reward()``).
  A run where you cannot say how much each term contributed is a run you cannot
  debug - the book says this in Chapter 5 and this class enforces it in the shape.
* **The trace is a list of segments, not a string.**  A segment is one contiguous
  span produced by one actor (``system``, ``user``, ``model``, ``tool``).  The
  loss mask is a function of these roles, so keeping the roles is what makes the
  mask checkable.
* **``mask`` is recorded, not derived.**  The writer records what its pipeline
  actually masked.  ``common.checks`` then verifies it against the segments.  A
  mask that is recomputed at check time can never be wrong, which would make the
  check worthless.
* **Provenance is part of the row.**  ``env_version``, ``config_hash`` and
  ``code_sha`` travel with every rollout, because a reward number without them
  cannot be reproduced by anyone else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "model", "tool"]

#: The roles a segment may have.  ``model`` is the only one the policy produced,
#: which is why it is the only one that contributes to the loss.
ROLES: tuple[str, ...] = ("system", "user", "model", "tool")


@dataclass(frozen=True)
class Segment:
    """One contiguous span of a trace, produced by one actor.

    ``tokens`` is a count, not a token list.  The fixtures in this repo are
    synthetic, so storing counts keeps them readable and diffable; a real pipeline
    stores the same field after tokenising and records the tokeniser version in
    ``Rollout.env_version``.
    """

    role: Role
    text: str
    tokens: int

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")
        if self.tokens < 0:
            raise ValueError(f"tokens must not be negative, got {self.tokens}")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(role=data["role"], text=data.get("text", ""), tokens=int(data["tokens"]))


@dataclass
class Rollout:
    """One sampled attempt at one prompt, with everything needed to reason about it.

    ``group_id`` is what makes a GRPO group: rollouts sharing a ``group_id`` are
    compared against each other (Chapter 4).  ``sample_index`` distinguishes them,
    and stays the same across rounds - which is what the round-stability check
    relies on.
    """

    episode_id: str
    prompt_id: str
    sample_index: int
    group_id: str
    segments: list[Segment]
    reward_terms: dict[str, float]
    env_version: str = "unknown"
    config_hash: str = "unknown"
    code_sha: str = "unknown"
    mask: list[int] | None = None
    turn_count: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- derived

    @property
    def total_tokens(self) -> int:
        """Token count of the whole trace, including everything the policy did not write."""
        return sum(segment.tokens for segment in self.segments)

    @property
    def model_tokens(self) -> int:
        """Tokens the policy produced - the only ones a loss may be computed on."""
        return sum(segment.tokens for segment in self.segments if segment.role == "model")

    def reward(self) -> float:
        """The fused reward: the sum of the terms.

        Summing is the default because the fusion policy (weights, gates, vetoes)
        is applied *before* the row is written, in ``common.fusion``.  By the time
        a row exists, its terms already are the contribution of each source.
        """
        return sum(self.reward_terms.values())

    def is_trainable(self) -> bool:
        """A row without a mask cannot be trained on, and must not be silently repaired."""
        return self.mask is not None and len(self.mask) == self.total_tokens

    # ------------------------------------------------------------ (de)serialise

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "prompt_id": self.prompt_id,
            "sample_index": self.sample_index,
            "group_id": self.group_id,
            "segments": [segment.to_dict() for segment in self.segments],
            "reward_terms": dict(self.reward_terms),
            "env_version": self.env_version,
            "config_hash": self.config_hash,
            "code_sha": self.code_sha,
            "mask": list(self.mask) if self.mask is not None else None,
            "turn_count": self.turn_count,
            "extras": dict(self.extras),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rollout":
        return cls(
            episode_id=data["episode_id"],
            prompt_id=data["prompt_id"],
            sample_index=int(data["sample_index"]),
            group_id=data["group_id"],
            segments=[Segment.from_dict(s) for s in data["segments"]],
            reward_terms={k: float(v) for k, v in data.get("reward_terms", {}).items()},
            env_version=data.get("env_version", "unknown"),
            config_hash=data.get("config_hash", "unknown"),
            code_sha=data.get("code_sha", "unknown"),
            mask=[int(v) for v in data["mask"]] if data.get("mask") is not None else None,
            turn_count=data.get("turn_count"),
            extras=dict(data.get("extras", {})),
        )

    @classmethod
    def from_json(cls, line: str) -> "Rollout":
        return cls.from_dict(json.loads(line))


def read_rollouts(path: str) -> list[Rollout]:
    """Read a JSONL file of rollouts, skipping blank lines."""
    rows: list[Rollout] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(Rollout.from_json(stripped))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: not a valid rollout row: {exc}") from exc
    return rows


def write_rollouts(path: str, rows: list[Rollout]) -> None:
    """Write rollouts as JSONL, one row per line, keys sorted for stable diffs."""
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.to_json())
            handle.write("\n")
