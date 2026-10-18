"""The last step of the credit chain: turn weights and log-probabilities into a scalar.

The loop already decides *what a turn is worth* (``train.turn_weights``) and *how a group is
centred* (``train.group_advantages``). What was missing is the step that turns those weights into
something a model can descend:

    loss = - mean over weighted turns t of ( w_t * mean over tokens j of log p(token_j) )

Three properties, each of them a test rather than a comment:

* **A zero weight contributes exactly nothing.** A turn the credit machinery called uninformative
  must not reach the gradient through a token count.
* **A turn is scale-free in its length.** A long turn and a short turn with the same mean
  log-probability contribute the same amount, because a turn that happens to be long must not
  outweigh a turn that happens to be short - the length bias Chapter 6 opens with.
* **The sign trains.** Raising a positively-weighted token's probability lowers the loss.

Nothing here imports torch. The container does the forward pass; this file does the arithmetic, so
the arithmetic can be tested on a laptop with no GPU and no weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TurnObjective:
    """One turn's contribution: how much it is worth, and what the model said about it."""

    weight: float
    logprobs: tuple[float, ...]

    @property
    def counted(self) -> bool:
        return self.weight != 0.0 and bool(self.logprobs)

    @property
    def contribution(self) -> float:
        if not self.counted:
            return 0.0
        return -self.weight * (sum(self.logprobs) / len(self.logprobs))


def objective(turns: Sequence[TurnObjective]) -> tuple[float, int]:
    """The round's scalar and the number of turns that carried a weight.

    Averaging over weighted turns rather than over turns keeps a group whose attempts produced
    different numbers of turns comparable: the count is reported, so a round that dropped most of
    its turns cannot look like a round that used them well.
    """
    counted = [turn for turn in turns if turn.counted]
    if not counted:
        return 0.0, 0
    return sum(turn.contribution for turn in counted) / len(counted), len(counted)


def distinct_weighted_turns(turns: Sequence[TurnObjective]) -> int:
    """How many turns had a non-zero weight - the sample of turns the update actually used."""
    return sum(1 for turn in turns if turn.counted)
