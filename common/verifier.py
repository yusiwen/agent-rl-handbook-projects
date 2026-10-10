"""Verifiers: who decides whether an attempt was any good, and what they can reach.

Chapter 5's first boundary is *coverage*.  A verifier suite is a machine that turns
attempts into numbers, and the set of tasks it can score is the ceiling on what you
can train: a task with no verifier is a task you cannot learn from, no matter how
large the model is.

So this module carries three things:

* ``Outcome`` - what a verifier returns: a pass/fail, a score, and the *terms* it
  contributed.  Terms are separate because the fused reward is computed later, and
  because "the rule said 1.0 and the judge said -0.2" is the diagnostic you want.
* ``Verifier`` - the protocol.  Two methods, both required: ``can_verify`` (coverage)
  and ``verify`` (judgement).  Keeping them apart is what makes coverage reportable.
* ``VerifierRegistry`` - dispatch by the task's ``verifier`` field, plus the coverage
  report.  An unregistered name raises ``MissingVerifier`` loudly: the failure mode to
  avoid is a task that silently scores zero and looks like a hard task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

#: The field a task uses to name the verifier that should score it.
TASK_VERIFIER_FIELD = "verifier"


@dataclass(frozen=True)
class Outcome:
    """The result of verifying one attempt.

    ``score`` is expected in ``[-1, 1]`` by convention - not because the framework
    requires it, but because a bounded score keeps one term from dominating a fusion
    total by accident.  The bound is *checked*, not clamped: silently rescaling a
    reward is how a pipeline hides a bug.
    """

    passed: bool
    score: float
    detail: str = ""
    terms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score {self.score!r} is outside [-1, 1]; rescale it deliberately")

    def as_terms(self, prefix: str) -> dict[str, float]:
        """Flatten into reward terms, e.g. ``{"rule": 1.0}``."""
        merged = {f"{prefix}/{key}": value for key, value in self.terms.items()}
        merged.setdefault(prefix, self.score)
        return merged


@runtime_checkable
class Verifier(Protocol):
    """What every verifier must provide.

    ``name`` must be unique in a registry.  ``can_verify`` answers "is this task in my
    coverage", and must not raise on a task of the wrong shape - it is asked about
    every task, including ones it will decline.
    """

    name: str

    def can_verify(self, task: Mapping[str, Any]) -> bool:  # pragma: no cover - protocol
        ...

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:  # pragma: no cover
        ...


class MissingVerifier(LookupError):
    """Raised when a task names a verifier that is not registered.

    This is deliberately an error and not a zero score: an unscored task and a failed
    task look identical in a reward curve, and only one of them is the reader's fault.
    """


class VerifierRegistry:
    """Name -> verifier, with a coverage report over a task set."""

    def __init__(self, verifiers: Iterable[Verifier] = ()) -> None:
        self._by_name: dict[str, Verifier] = {}
        for verifier in verifiers:
            self.register(verifier)

    # ------------------------------------------------------------------ registry

    def register(self, verifier: Verifier) -> None:
        name = getattr(verifier, "name", "")
        if not name or not isinstance(name, str):
            raise ValueError("a verifier must have a non-empty string 'name'")
        if name in self._by_name:
            raise ValueError(f"verifier {name!r} is already registered")
        if not callable(getattr(verifier, "verify", None)):
            raise ValueError(f"verifier {name!r} has no callable verify()")
        if not callable(getattr(verifier, "can_verify", None)):
            raise ValueError(f"verifier {name!r} has no callable can_verify()")
        self._by_name[name] = verifier

    @property
    def names(self) -> list[str]:
        return sorted(self._by_name)

    def get(self, name: str) -> Verifier:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise MissingVerifier(
                f"no verifier named {name!r}; registered: {self.names or 'none'}"
            ) from exc

    def for_task(self, task: Mapping[str, Any]) -> Verifier:
        """The verifier a task asked for, by its ``verifier`` field."""
        name = task.get(TASK_VERIFIER_FIELD)
        if not isinstance(name, str) or not name:
            raise MissingVerifier(f"task {task.get('id', '<no id>')!r} names no verifier")
        return self.get(name)

    # ------------------------------------------------------------------ coverage

    def coverage(self, tasks: Iterable[Mapping[str, Any]]) -> "CoverageReport":
        """How much of a task set this suite can actually score.

        Counted, not estimated.  Chapter 5 asks the reader to report this number
        before training, because it is an upper bound on what training can add.
        """
        total = 0
        named_missing: list[str] = []
        declared_but_blind: list[str] = []
        by_verifier: dict[str, int] = {}
        for task in tasks:
            total += 1
            task_id = str(task.get("id", "<no id>"))
            name = task.get(TASK_VERIFIER_FIELD)
            if not isinstance(name, str) or not name:
                named_missing.append(task_id)
                continue
            verifier = self._by_name.get(name)
            if verifier is None:
                named_missing.append(f"{task_id} (wants {name!r})")
                continue
            if not verifier.can_verify(task):
                declared_but_blind.append(f"{task_id} (declares {name!r}, which declined it)")
                continue
            by_verifier[name] = by_verifier.get(name, 0) + 1
        covered = total - len(named_missing) - len(declared_but_blind)
        return CoverageReport(
            total=total,
            covered=covered,
            by_verifier=by_verifier,
            no_verifier=named_missing,
            declined=declared_but_blind,
        )


@dataclass(frozen=True)
class CoverageReport:
    """The answer to "how many of my tasks can anything score". """

    total: int
    covered: int
    by_verifier: dict[str, int]
    no_verifier: list[str]
    declined: list[str]

    @property
    def fraction(self) -> float:
        return 0.0 if self.total == 0 else self.covered / self.total

    def render(self) -> str:
        lines = [
            f"coverage: {self.covered}/{self.total} tasks ({self.fraction:.0%})",
            "  by verifier: "
            + (", ".join(f"{name}={count}" for name, count in sorted(self.by_verifier.items())) or "-"),
        ]
        if self.no_verifier:
            lines.append(f"  no verifier ({len(self.no_verifier)}): {', '.join(self.no_verifier[:5])}"
                         + (" ..." if len(self.no_verifier) > 5 else ""))
        if self.declined:
            lines.append(f"  declined ({len(self.declined)}): {', '.join(self.declined[:5])}"
                         + (" ..." if len(self.declined) > 5 else ""))
        if not self.no_verifier and not self.declined:
            lines.append("  nothing uncovered - every task can be scored")
        return "\n".join(lines)
