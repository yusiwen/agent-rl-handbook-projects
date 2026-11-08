"""The library, and the gate that decides what may enter it.

A skill is a procedure with provenance: which failures produced it, and what it was shown to fix.
The gate is deliberately behavioural rather than editorial - a candidate is admitted only if a
policy that is given the skill answers more held-out instances of its family correctly than the
same policy without it. Text that *sounds* like a fix does not enter; a decoy that is plausible
and wrong is rejected by construction, because following it makes the score worse.

The comparison is made on held-out instances of the family, never on the instances that produced
the failure: a skill that only repairs the three examples it was written from is memorisation with
extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .families import Family, Instance, score
from .retrieval import SkillDoc

Policy = Callable[[str, str], str]
"""policy(ask, injected_procedure) -> answer. Empty injected_procedure means the baseline."""


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    trigger: str
    procedure: str
    source: str = "failure-trace"

    def doc(self) -> SkillDoc:
        return SkillDoc(name=self.name, family=self.family, trigger=self.trigger, procedure=self.procedure)


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    instances: int
    baseline: int
    with_skill: int
    accepted: bool
    reason: str

    @property
    def gain(self) -> int:
        return self.with_skill - self.baseline


def validate(
    candidate: Candidate,
    family: Family,
    policy: Policy,
    holdout: Sequence[Instance],
    minimum_gain: int = 1,
) -> Verdict:
    """Run the same policy twice over the same held-out instances, with and without the text."""
    holdout = [instance for instance in holdout if instance.family == family.id]
    if not holdout:
        return Verdict(candidate, 0, 0, 0, False, "no held-out instances for this family")

    baseline = sum(1 for instance in holdout if score(instance, policy(instance.ask, "")))
    with_skill = sum(1 for instance in holdout if score(instance, policy(instance.ask, candidate.procedure)))
    gain = with_skill - baseline

    if gain < minimum_gain:
        reason = f"no gain: {with_skill} of {len(holdout)} with the skill, {baseline} without"
        return Verdict(candidate, len(holdout), baseline, with_skill, False, reason)
    if with_skill < baseline:
        return Verdict(candidate, len(holdout), baseline, with_skill, False, "the skill made it worse")
    return Verdict(
        candidate,
        len(holdout),
        baseline,
        with_skill,
        True,
        f"admitted: {with_skill} of {len(holdout)} with the skill, {baseline} without",
    )


@dataclass
class Library:
    """One skill per family. A second candidate for the same family replaces it only if it wins."""

    skills: dict[str, SkillDoc] = field(default_factory=dict)
    evidence: dict[str, Verdict] = field(default_factory=dict)
    rejected: list[Verdict] = field(default_factory=list)

    def admit(self, verdict: Verdict) -> bool:
        if not verdict.accepted:
            self.rejected.append(verdict)
            return False
        previous = self.evidence.get(verdict.candidate.family)
        if previous is not None and previous.with_skill >= verdict.with_skill:
            # the incumbent is at least as good on its own held-out set; keeping it is the
            # conservative choice, and it stops a lucky candidate from churning the library
            self.rejected.append(verdict)
            return False
        self.skills[verdict.candidate.family] = verdict.candidate.doc()
        self.evidence[verdict.candidate.family] = verdict
        return True

    def docs(self) -> list[SkillDoc]:
        return list(self.skills.values())

    def procedure_for(self, family: str) -> str:
        skill = self.skills.get(family)
        return skill.procedure if skill else ""

    def injection(self, *families: str) -> str:
        """The text a router would prepend for the given families, in a stable order."""
        parts = [self.skills[family].procedure for family in families if family in self.skills]
        if not parts:
            return ""
        return "Procedure to follow:\n" + "\n".join(f"- {part}" for part in parts)

    def __len__(self) -> int:
        return len(self.skills)
