"""Group-relative advantages, and the two ways Chapter 6 turns a score into a weight.

GRPO, in one function: for each group (the samples of one prompt), subtract the group's mean and
divide by its spread.  A flat group - every attempt scored the same - produces zeros, which is the
honest answer: there is nothing in that group to learn from, and inventing a gradient from noise is
how a training run "works" while learning nothing.

The second function is this chapter's own question, not GRPO's: *where* does a turn's weight come
from?  Two modes, both written down rather than implied:

* ``trajectory`` - every turn of an attempt carries that attempt's group-relative advantage.  This
  is plain GRPO with a multi-turn trace: the outcome decides, and per-turn scores are logged for
  diagnosis.  It is the default, because it is the only mode whose credit is *grounded*.
* ``turn`` - the attempt's advantage plus a shaped term from the turn scores, standardised within
  the attempt.  This is Chapter 6, Lesson 2's "trajectory outcome + turn-level shaping": the
  outcome still decides the sign, and the turn scores redistribute weight *inside* one attempt.
  The shaping coefficient is a constant here, so that a reader can see exactly how much of the
  gradient came from the judge rather than the verifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .agent import Episode

#: Spreads below this are treated as flat: reward terms are floats and exact equality is a
#: coincidence, not a measurement (the same epsilon `common.runs` uses).
EPS = 1e-9

#: How much of a turn's weight may come from the critic's turn scores, in ``turn`` mode.  Half is a
#: deliberate choice: the outcome keeps the sign, the judge only redistributes.
TURN_SHAPING = 0.5

CREDIT_MODES = ("trajectory", "turn")


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Group-relative advantages: mean-centred, spread-normalised, zeros for a flat group."""
    if not rewards:
        raise ValueError("cannot compute advantages for an empty group")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    spread = math.sqrt(variance)
    if spread <= EPS:
        return [0.0] * len(rewards)
    return [(value - mean) / spread for value in rewards]


def standardise(values: Sequence[float]) -> list[float]:
    """Mean-centre and spread-normalise one trajectory's turn scores; zeros if they are all equal."""
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    spread = math.sqrt(variance)
    if spread <= EPS:
        return [0.0] * len(values)
    return [(value - mean) / spread for value in values]


def turn_weights(
    episode: Episode,
    advantage: float,
    *,
    mode: str = "trajectory",
    turn_scores: Sequence[float] | None = None,
    shaping: float = TURN_SHAPING,
) -> list[float]:
    """One weight per turn, ready for ``agent.policy_updates``."""
    if mode not in CREDIT_MODES:
        raise ValueError(f"unknown credit mode {mode!r}; expected one of {CREDIT_MODES}")
    count = len(episode.actions)
    if mode == "trajectory":
        return [advantage] * count
    if turn_scores is None:
        raise ValueError("turn mode needs the critic's per-turn scores")
    if len(turn_scores) != count:
        raise ValueError(f"{len(turn_scores)} turn score(s) for {count} turn(s)")
    shaped = standardise(turn_scores)
    return [advantage + shaping * value for value in shaped]


@dataclass(frozen=True)
class GroupSummary:
    """What one group's numbers were, so a round can be read without the rows."""

    group_id: str
    rewards: list[float]
    advantages: list[float]

    @property
    def flat(self) -> bool:
        return all(abs(value) <= EPS for value in self.advantages)


def summarise_groups(episodes: Sequence[Episode], totals: Mapping[str, float]) -> list[GroupSummary]:
    """Group the attempts, compute each group's advantages, and keep the pairing visible."""
    grouped: dict[str, list[tuple[float, str]]] = {}
    for episode in episodes:
        total = totals[episode.rollout.episode_id]
        grouped.setdefault(episode.rollout.group_id, []).append((total, episode.rollout.episode_id))

    summaries: list[GroupSummary] = []
    for group_id in sorted(grouped):
        rewards = [total for total, _ in grouped[group_id]]
        advantages = group_advantages(rewards)
        summaries.append(GroupSummary(group_id=group_id, rewards=rewards, advantages=advantages))
    return summaries


def advantage_for(summaries: Sequence[GroupSummary], episode: Episode, totals: Mapping[str, float]) -> float:
    """The advantage of one episode, looked up from its group's summary."""
    for summary in summaries:
        if summary.group_id != episode.rollout.group_id:
            continue
        rewards = summary.rewards
        if len(rewards) == 1:
            return 0.0  # a group of one has no relative information at all
        return group_advantages(rewards)[rewards.index(totals[episode.rollout.episode_id])]
    raise KeyError(f"no summary for group {episode.rollout.group_id!r}")
