"""Fusion: turning the terms of a reward into the one number the trainer consumes.

Chapter 5's rules, in code:

* **Terms are recorded, never collapsed silently.**  ``fuse`` returns the total *and*
  the per-term contributions, and the rollout row keeps the contributions.  A fusion
  step that only returns a float makes the next bug unfalsifiable.
* **Weights are explicit.**  There is no "sensible default" that guesses how much the
  rule matters against the judge.  A missing weight is an error.
* **Conflicts are a signal, not noise.**  When a rule term and a judge term disagree in
  sign on the same attempt, that attempt is the most informative one in the batch: it
  is either a bad judge or a checker that is too strict.  ``detect_conflicts`` finds
  these so they can be read, instead of averaging them away.
* **A veto is a policy, not a hack.**  Some rules are non-negotiable (a security
  policy, an invalid output format).  ``veto`` expresses that honestly, by zeroing the
  total and saying so - rather than by giving the rule a huge weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: The three fusion policies the book names.  Each is a different answer to
#: "what happens when the sources disagree".
MODES = ("weighted", "gated", "veto")


@dataclass(frozen=True)
class FusionPolicy:
    """How to combine terms.

    ``weights`` maps a term name to its weight; every term that should reach the total
    must appear here, so that a new term cannot sneak in unweighted.

    ``mode``:

    * ``weighted`` - the default: ``sum(weight[t] * terms[t])``.
    * ``gated`` - the term named by ``gate`` decides whether the term named by
      ``gated`` is counted at all.  Use when one source is authoritative and the other
      is a tie-breaker.
    * ``veto`` - if any term named in ``veto_on`` is negative, the total is zero.  Use
      for constraints that cannot be averaged away.
    """

    weights: Mapping[str, float]
    mode: str = "weighted"
    gate: str | None = None
    gated: str | None = None
    veto_on: Sequence[str] = ()
    gate_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown fusion mode {self.mode!r}; expected one of {MODES}")
        if not self.weights:
            raise ValueError("a fusion policy needs at least one weight")
        if self.mode == "gated" and (self.gate is None or self.gated is None):
            raise ValueError("gated fusion needs both 'gate' and 'gated'")
        if self.mode == "veto" and not self.veto_on:
            raise ValueError("veto fusion needs at least one term in 'veto_on'")


@dataclass(frozen=True)
class Fusion:
    """The result: a total, the contribution of every term, and any conflict found."""

    total: float
    contributions: dict[str, float]
    conflicts: list[str] = field(default_factory=list)
    vetoed_by: str | None = None

    def render(self) -> str:
        parts = ", ".join(f"{name}={value:+.3f}" for name, value in sorted(self.contributions.items()))
        line = f"total={self.total:+.3f}  ({parts})"
        if self.vetoed_by is not None:
            line += f"  [vetoed by {self.vetoed_by}]"
        for conflict in self.conflicts:
            line += f"\n  conflict: {conflict}"
        return line


def fuse(terms: Mapping[str, float], policy: FusionPolicy) -> Fusion:
    """Combine reward terms into a total, keeping every contribution traceable.

    Raises ``KeyError`` when a term has no weight and ``ValueError`` when the policy
    refers to a term that was not supplied - both are mistakes worth stopping for.
    """
    missing = sorted(set(terms) - set(policy.weights))
    if missing:
        raise KeyError(
            f"no weight for term(s) {missing}; add them to the policy rather than "
            "letting an unweighted term reach the total"
        )

    contributions: dict[str, float] = {}
    for name, policy_weight in policy.weights.items():
        if name not in terms:
            raise ValueError(f"policy weights {name!r} but no such term was supplied")

    gate_open = True
    if policy.mode == "gated":
        assert policy.gate is not None and policy.gated is not None  # checked in __post_init__
        if policy.gate not in terms:
            raise ValueError(f"policy gates on {policy.gate!r} but no such term was supplied")
        gate_open = terms[policy.gate] >= policy.gate_threshold

    for name, policy_weight in policy.weights.items():
        if policy.mode == "gated" and name == policy.gated and not gate_open:
            contributions[name] = 0.0
            continue
        contributions[name] = policy_weight * terms[name]

    total = sum(contributions.values())

    vetoed_by: str | None = None
    if policy.mode == "veto":
        for name in policy.veto_on:
            if name not in terms:
                raise ValueError(f"policy vetoes on {name!r} but no such term was supplied")
            if terms[name] < 0:
                vetoed_by = name
                total = 0.0
                break

    return Fusion(total=total, contributions=contributions, conflicts=[], vetoed_by=vetoed_by)


def detect_conflicts(
    terms: Mapping[str, float],
    sources: Mapping[str, Iterable[str]],
    margin: float = 0.2,
) -> list[str]:
    """Report disagreements between sources that are large enough to act on.

    ``sources`` names the term groups, e.g. ``{"rule": ["exact"], "judge": ["judge"]}``.
    A conflict is two terms from *different* groups with opposite signs and both at
    least ``margin`` in magnitude.  Small disagreements are normal and are ignored.
    """
    if margin < 0:
        raise ValueError("margin must not be negative")
    conflicts: list[str] = []
    groups = list(sources.items())
    for index, (left_name, left_terms) in enumerate(groups):
        for right_name, right_terms in groups[index + 1 :]:
            for left in left_terms:
                for right in right_terms:
                    if left not in terms or right not in terms:
                        continue
                    left_value, right_value = terms[left], terms[right]
                    if abs(left_value) < margin or abs(right_value) < margin:
                        continue
                    if (left_value > 0) != (right_value > 0):
                        conflicts.append(
                            f"{left_name}[{left}]={left_value:+.3f} disagrees with "
                            f"{right_name}[{right}]={right_value:+.3f}"
                        )
    return conflicts


def fuse_with_conflicts(
    terms: Mapping[str, float],
    policy: FusionPolicy,
    sources: Mapping[str, Iterable[str]],
    margin: float = 0.2,
) -> Fusion:
    """``fuse`` plus the conflict report, for pipelines that log both."""
    result = fuse(terms, policy)
    conflicts = detect_conflicts(terms, sources, margin=margin)
    return Fusion(
        total=result.total,
        contributions=result.contributions,
        conflicts=conflicts,
        vetoed_by=result.vetoed_by,
    )
