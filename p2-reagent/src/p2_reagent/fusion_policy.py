"""The fusion point: rule reward and critic reward in, one scalar out - and the weights written down.

Chapter 6, Lesson 5 puts this in one file on purpose: a reward you cannot point at is a reward you
cannot debug, and a weight nobody wrote down is a decision nobody made.  So the weights are
constants here, the conflict policy is a string next to them, and ``fuse_terms`` is the only
function a trainer is allowed to call.

Two stages, from the chapter's own sequencing rule ("get the loop right before the signal"):

* ``rule-only`` - the fusion point is in place and returns only the rule terms.  If the loop does
  not close here, the problem is plumbing; if it closes and then breaks when the critic is added,
  the problem is the signal.
* ``full`` - the rule's outcome and turn terms plus the critic's score.

The conflict policy, stated in one sentence rather than implied by a number:

    The outcome term is the truth and the critic is a tie-breaker.  When the rule says the attempt
    failed and the critic scores its *process* above the midpoint, the critic's contribution is
    halved and the pair is logged as a conflict: the attempt is worth reading - it is where the
    judge's blind spot shows - but a good process that did not reach the outcome is not a success.
    Averaging that away is how a reward model teaches a policy to look right instead of be right.

That is why the mode is ``weighted`` rather than ``veto``: a veto would throw away the critic's
signal on exactly the failed attempts the loop most needs to learn from, and Chapter 6's density
argument dies with it.
"""

from __future__ import annotations

from typing import Mapping

from common.fusion import Fusion, FusionPolicy, fuse_with_conflicts

#: Which terms each stage fuses.  A term that is supplied but not weighted is an error in
#: ``common.fusion``, and that is deliberate: a stage cannot silently ignore a signal.
RULE_ONLY = "rule-only"
FULL = "full"
STAGES = (RULE_ONLY, FULL)

WEIGHTS: dict[str, dict[str, float]] = {
    # The turn term is *not* zero here, and that is a measured decision rather than a taste: with
    # outcome alone, a cold policy that solves nothing produces an all-zero reward, every group is
    # flat, the advantage is zero and the update is a no-op - the loop cannot start.  The turn term
    # is what gives the first rounds a gradient, and it is the environment's own progress signal
    # (how close to the goal the attempt got), so no model is involved.  It is a *maximum* over the
    # attempt rather than a mean over turns: a mean lets a policy pad with neutral turns and
    # outscore an attempt that took the right action and reported a wrong answer.
    RULE_ONLY: {"rule_outcome": 0.6, "rule_turn": 0.4},
    FULL: {"rule_outcome": 0.5, "rule_turn": 0.3, "critic": 0.2},
}

#: The rule's terms, and the critic's.  Used for conflict detection, and logged with every row.
SOURCES = {"rule": ("rule_outcome", "rule_turn"), "critic": ("critic",)}

#: The one sentence a reader needs, above.  Kept here rather than in a README so that a change to
#: the weights has to be made in the same place as a change to the policy.
CONFLICT_POLICY = (
    "the outcome term is the truth and the critic is a tie-breaker: when the rule says the attempt "
    "failed and the critic scores the process above 0.5, the critic's contribution is halved and "
    "the pair is logged as a conflict"
)

#: How much of the critic's contribution survives a rule-vs-critic conflict.
CONFLICT_DAMPING = 0.5

#: A conflict this small is noise, not a finding.  Chapter 5 uses the same idea one level up.
CONFLICT_MARGIN = 0.25


def policy_for(stage: str) -> FusionPolicy:
    """The fusion policy for a stage, with its weights checked at construction time."""
    if stage not in WEIGHTS:
        raise KeyError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return FusionPolicy(weights=WEIGHTS[stage], mode="weighted")


def terms_for(stage: str, *, rule_outcome: float, rule_turn: float, critic: float | None) -> dict[str, float]:
    """Exactly the terms this stage fuses.  Supplying the wrong set is an error, by design."""
    weights = WEIGHTS[stage]
    terms = {"rule_outcome": float(rule_outcome), "rule_turn": float(rule_turn)}
    if "critic" in weights:
        if critic is None:
            raise ValueError(f"stage {stage!r} weights a critic term but none was supplied")
        terms["critic"] = float(critic)
    elif critic is not None:
        raise ValueError(
            f"stage {stage!r} does not weight a critic term, but a critic score was supplied; "
            "that is the rule-only stage and it must be able to run with no judge at all"
        )
    return terms


def fuse_terms(terms: Mapping[str, float], stage: str) -> Fusion:
    """Fuse the terms for a stage, applying the written conflict policy.

    The returned ``contributions`` are what belongs in a rollout row's ``reward_terms``: their sum
    is the number the trainer consumes, and each name still says which source produced it.  The
    *raw* scores stay in the row's extras, so a damped contribution can always be traced back to
    the pair that produced it.
    """
    policy = policy_for(stage)
    result = fuse_with_conflicts(terms, policy, sources=SOURCES, margin=CONFLICT_MARGIN)

    if _failed_outcome_with_good_process(terms, stage):
        damped = dict(terms)
        damped["critic"] = terms["critic"] * CONFLICT_DAMPING
        result = fuse_with_conflicts(damped, policy, sources=SOURCES, margin=CONFLICT_MARGIN)
        return Fusion(
            total=result.total,
            contributions=result.contributions,
            conflicts=list(result.conflicts)
            + ["outcome=0 with a critic score above the midpoint: critic contribution halved"],
            vetoed_by=None,
        )
    return result


def _failed_outcome_with_good_process(terms: Mapping[str, float], stage: str) -> bool:
    if "critic" not in WEIGHTS[stage] or "critic" not in terms:
        return False
    return terms.get("rule_outcome", 0.0) <= 0.0 and terms["critic"] > 0.5
