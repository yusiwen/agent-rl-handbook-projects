"""A tabular softmax policy, so the credit machinery can be studied without a neural network.

Why a table, and why this is not a shortcut.  Chapter 6 is about the *signal*: which turn gets
blame, what it costs to see it, and whether feedback reaches the next round of sampling.  A
neural policy would add an optimiser, a random initialisation and a dozen hyperparameters
between the signal and the evidence.  A table does not: its log-probabilities are exact, its
gradient is closed-form, and its entropy is a number you can watch.  Every claim this project
makes about credit assignment can therefore be tested offline, in a second, with no GPU.

The maths, written down because the chapter asks the reader to be able to write it down:

    pi(a | s)       = exp(theta[s, a] / T) / sum_b exp(theta[s, b] / T)
    grad log pi(a|s) = onehot(a) - pi(. | s)
    theta[s, a]    += learning_rate * weight * grad log pi(a | s)

With ``weight`` set to the group-relative advantage, the last line *is* GRPO's policy gradient
for a softmax policy: no critic, no value baseline, one group of samples per prompt.  The
``weight`` may instead be a turn-level weight (Chapter 6's shaping term); the update does not
care where the number came from, which is exactly the separation the project needs.

What the table cannot do, stated here so nobody discovers it later: it cannot generalise from
one task to another.  Two tasks that share a state but want different actions need different
keys, which is why the key includes the task id.  A real policy generalises through the prompt;
a table does not, and pretending otherwise would make the demo lie about what it measured.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: Below this probability an action is still sampleable, but the guard keeps the arithmetic
#: away from log(0) when a table has been pushed hard by a few updates.
MIN_PROBABILITY = 1e-9


@dataclass(frozen=True)
class Update:
    """One action taken, and the weight the update should give it.

    ``menu`` is recorded because the gradient is relative to the alternatives that were on
    offer: the same action chosen from a menu of two and a menu of seven carries different
    information, and a table that forgot the menu could not tell them apart.
    """

    signature: str
    menu: tuple[str, ...]
    action: str
    weight: float

    def __post_init__(self) -> None:
        if self.action not in self.menu:
            raise ValueError(f"action {self.action!r} is not in its own menu")
        if not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight!r}")


class TabularPolicy:
    """A softmax over actions, keyed by a state signature."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        learning_rate: float = 1.0,
        version: str = "policy-v0",
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.version = version
        self.seed = seed
        self._rng = random.Random(seed)
        self._theta: dict[str, dict[str, float]] = {}
        self._updates = 0

    # ------------------------------------------------------------------ reading it
    def _row(self, signature: str, menu: Sequence[str]) -> dict[str, float]:
        """The row for this signature, created at zero (uniform) if it is new."""
        row = self._theta.setdefault(signature, {})
        for action in menu:
            row.setdefault(action, 0.0)
        return row

    def probs(self, signature: str, menu: Sequence[str]) -> list[float]:
        """The probability of each action in ``menu``, in the menu's order."""
        if not menu:
            raise ValueError("cannot score an empty menu")
        row = self._row(signature, menu)
        scaled = [row[action] / self.temperature for action in menu]
        top = max(scaled)
        weights = [math.exp(value - top) for value in scaled]
        total = sum(weights)
        return [max(weight / total, MIN_PROBABILITY) for weight in weights]

    def log_prob(self, signature: str, menu: Sequence[str], action: str) -> float:
        if action not in menu:
            raise ValueError(f"action {action!r} is not in the menu")
        return math.log(self.probs(signature, menu)[menu.index(action)])

    def entropy(self, signature: str, menu: Sequence[str]) -> float:
        """Shannon entropy in nats: ``log(len(menu))`` when uniform, towards 0 when collapsed."""
        return -sum(p * math.log(p) for p in self.probs(signature, menu))

    def greedy(self, signature: str, menu: Sequence[str]) -> str:
        """The most likely action, for inspection rather than for sampling."""
        probabilities = self.probs(signature, menu)
        return menu[max(range(len(menu)), key=lambda index: probabilities[index])]

    # ------------------------------------------------------------------ sampling
    def sample(self, signature: str, menu: Sequence[str]) -> str:
        """One draw from the policy.  Seeded, so a run is reproducible from its header."""
        probabilities = self.probs(signature, menu)
        threshold = self._rng.random()
        cumulative = 0.0
        for action, probability in zip(menu, probabilities):
            cumulative += probability
            if threshold <= cumulative:
                return action
        return menu[-1]

    # ------------------------------------------------------------------ updating
    def apply(self, updates: Iterable[Update]) -> None:
        """One gradient step per update, at ``learning_rate``.

        Two deliberate choices.  Updates are applied one at a time rather than summed first, so
        that a caller can inspect the table after a single sample - the loop logs per round, and a
        batch-summed gradient would hide a wrong weight until the next round.  And there is no
        gradient clipping: the step is the learning rate times the weight, nothing else, because a
        table with a runaway row is a *visible* problem (the entropy falls off a cliff and the
        check fires) rather than a silent one.  A neural policy would need the guard; a table can
        show you the damage.
        """
        for update in updates:
            probabilities = self.probs(update.signature, update.menu)
            row = self._row(update.signature, update.menu)
            for action, probability in zip(update.menu, probabilities):
                gradient = (1.0 if action == update.action else 0.0) - probability
                row[action] += self.learning_rate * update.weight * gradient
            self._updates += 1

    # ------------------------------------------------------------------ describing
    @property
    def updates_applied(self) -> int:
        return self._updates

    def table(self) -> dict[str, dict[str, float]]:
        """A copy of the parameters, for logging and for version-to-version comparison."""
        return {signature: dict(row) for signature, row in self._theta.items()}

    def snapshot(self) -> dict:
        """Everything needed to describe this policy in a run header."""
        return {
            "version": self.version,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "updates_applied": self._updates,
            "signatures": len(self._theta),
        }

    def bumped(self, version: str) -> "TabularPolicy":
        """A copy under a new version name, keeping the parameters and the RNG position.

        The loop uses this between rounds: round N+1 must sample from the policy round N
        produced, and the version string in the header is what makes that checkable rather than
        asserted.
        """
        clone = TabularPolicy(
            temperature=self.temperature,
            seed=self.seed,
            learning_rate=self.learning_rate,
            version=version,
        )
        clone._theta = self.table()
        clone._rng = self._rng
        clone._updates = self._updates
        return clone


def from_table(table: Mapping[str, Mapping[str, float]], **kwargs) -> TabularPolicy:
    """Rebuild a policy from a logged table (``--policy-table``), for replay."""
    policy = TabularPolicy(**kwargs)
    policy._theta = {signature: dict(row) for signature, row in table.items()}
    return policy
