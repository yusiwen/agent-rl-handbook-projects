"""Process feedback and turn-level credit: the loop Chapter 6 asks the reader to close.

The pass criterion this module exists to satisfy, in the chapter's words: *the training log shows
feedback reaching the next round of sampling - not just scores being recorded*.  Three mechanisms
make that checkable instead of asserted:

* **a policy version per round**, written into every rollout row, so a row that was sampled by an
  older policy can never be mistaken for a fresh one;
* **a digest of the parameters per round**, so the loop can prove the update changed something
  rather than only that the reward moved;
* **the same sample ids across rounds**, re-scored, so the reader can see a specific attempt's
  reward change when the policy that produced it changed.

The default stage is ``rule-only`` on purpose (Chapter 6, Lesson 5's sequencing rule): the fusion
point is in place, the loop is proven, and only then is a judge added.  A loop that fails here has
a plumbing bug; a loop that works here and breaks when the judge arrives has a signal bug, and
those two are worth keeping apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

from common.checks import DEFAULT_MIN_SPREAD_FRACTION, CheckResult, run_all
from common.runs import RunHeader, RunWriter, digest_of, round_record, spread_fraction
from common.schema import Rollout

from . import env, fusion_policy
from .agent import DEFAULT_MAX_TURNS, Episode, run_episode
from .critics import BudgetedCritic, FixtureCritic, ModelCritic, RuleCritic, Verdict, disagreement
from .policy import TabularPolicy
from .train import CREDIT_MODES, advantage_for, summarise_groups, turn_weights

PROJECT = "p2-reagent"

#: How far the mean entropy may fall below round 0's before the loop calls it collapse.  Chapter 6,
#: Lesson 4's Echo Trap is exactly this failure, and it is invisible in a reward curve.
MIN_ENTROPY_FRACTION = 0.5


def _digest_of_table(table: dict) -> str:
    payload = json.dumps(table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class Config:
    """Everything that decides a run, and nothing that does not."""

    tasks: tuple[str, ...] = tuple(task.task_id for task in env.all_tasks())
    samples: int = 4
    rounds: int = 3
    stage: str = fusion_policy.RULE_ONLY
    credit: str = "trajectory"
    learning_rate: float = 1.0
    temperature: float = 1.0
    seed: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    #: True when the policy is a remote endpoint rather than a table: there is no update to apply,
    #: so the lineage and entropy checks do not apply and are dropped with a reason at the call site.
    served: bool = False
    #: The share of groups that must show a non-zero reward difference.  The shared default suits
    #: a calibrated set; this project's 43 tasks are not calibrated against the cold policy, and
    #: the measured fraction is a difficulty number (see check_p2.py).  It travels in the config so
    #: a run record says which floor it was judged against.
    min_spread_fraction: float = DEFAULT_MIN_SPREAD_FRACTION

    def __post_init__(self) -> None:
        if self.samples < 2:
            raise ValueError("a GRPO group needs at least two samples")
        if self.rounds < 1:
            raise ValueError("a loop needs at least one round")
        if self.stage not in fusion_policy.STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.credit not in CREDIT_MODES:
            raise ValueError(f"unknown credit mode {self.credit!r}")
        if not self.tasks:
            raise ValueError("no tasks")

    def as_dict(self) -> dict:
        return {
            "tasks": list(self.tasks),
            "samples": self.samples,
            "rounds": self.rounds,
            "stage": self.stage,
            "credit": self.credit,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_turns": self.max_turns,
        }

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]

    def notes(self, critic_source: str) -> str:
        return (
            f"{len(self.tasks)} task(s) x {self.samples} sample(s), {self.rounds} round(s); "
            f"stage={self.stage}, credit={self.credit}, lr={self.learning_rate}, "
            f"temperature={self.temperature}, critic={critic_source}"
        )


@dataclass
class RoundResult:
    """One round: its rows, its verdicts, its checks, and what the update changed."""

    index: int
    policy_version: str
    table_digest: str
    episodes: list[Episode] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    rule_verdicts: dict[str, Verdict] = field(default_factory=dict)
    terms_raw: dict[str, dict[str, float]] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    gates: dict[str, str] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    updates_applied: int = 0

    @property
    def rollouts(self) -> list[Rollout]:
        return [episode.rollout for episode in self.episodes]

    @property
    def mean_reward(self) -> float:
        return sum(self.totals.values()) / len(self.totals) if self.totals else 0.0

    @property
    def solved_share(self) -> float:
        return (
            sum(1 for episode in self.episodes if episode.solved) / len(self.episodes)
            if self.episodes
            else 0.0
        )

    @property
    def spread(self) -> float:
        return spread_fraction(self.rollouts)

    @property
    def term_means(self) -> dict[str, float]:
        means: dict[str, float] = {}
        for episode in self.episodes:
            for name, value in episode.rollout.reward_terms.items():
                means[name] = means.get(name, 0.0) + value
        return {name: value / len(self.episodes) for name, value in means.items()} if self.episodes else {}

    def summary(self) -> dict:
        return {
            "round": self.index,
            "policy_version": self.policy_version,
            "table_digest": self.table_digest,
            "reward_mean": round(self.mean_reward, 4),
            "solved_share": round(self.solved_share, 4),
            "term_means": {name: round(value, 4) for name, value in sorted(self.term_means.items())},
            "spread": round(self.spread, 4),
            "entropy_before": round(self.entropy_before, 4),
            "entropy_after": round(self.entropy_after, 4),
            "conflicts": sum(len(items) for items in self.conflicts.values()),
            "updates_applied": self.updates_applied,
            "checks_ok": all(check.ok for check in self.checks),
        }


# --------------------------------------------------------------------------- checks


def check_entropy(results: Sequence[RoundResult], minimum_fraction: float = MIN_ENTROPY_FRACTION) -> CheckResult:
    """No round halves the entropy of the one before it: Chapter 6's Echo Trap, watched per round.

    A learning policy *does* lose entropy - that is what sharpening looks like - so comparing with
    round 0 would call every successful run a collapse.  The failure the chapter describes is a
    cliff: one round in which the distribution gives up most of its spread, after which sampling
    produces the same output regardless of the task.
    """
    if len(results) < 2:
        return CheckResult(
            name="entropy",
            what_it_checks="no round halves the entropy of the one before it",
            ok=True,
            messages=["one round: no cliff to measure"],
        )
    messages: list[str] = [
        "entropy per round: " + " -> ".join(f"{result.entropy_before:.2f}" for result in results)
        + f" -> {results[-1].entropy_after:.2f}"
    ]
    ok = True
    for previous, current in zip(results, results[1:]):
        if previous.entropy_after <= 0.0:
            continue
        fraction = current.entropy_before / previous.entropy_after
        if fraction < minimum_fraction:
            ok = False
            messages.append(
                f"round {current.index} started at {fraction:.0%} of round {previous.index}'s entropy"
            )
    if ok:
        messages.append("no round gave up more than half its entropy")
    return CheckResult(
        name="entropy",
        what_it_checks="no round halves the entropy of the one before it",
        ok=ok,
        messages=messages,
    )


def check_lineage(results: Sequence[RoundResult]) -> CheckResult:
    """Each round's rows came from that round's policy, and the update changed the parameters.

    Two failures, and they mean different things.  A version mismatch means rows from one policy
    were mixed into another round's batch.  An unchanged parameter digest means the update did
    nothing - a dashboard, not a loop - even if the rewards moved for sampling reasons.
    """
    messages: list[str] = []
    ok = True
    for previous, current in zip(results, results[1:]):
        if current.policy_version == previous.policy_version:
            ok = False
            messages.append(
                f"round {current.index} sampled under {current.policy_version}, the same version as "
                f"round {previous.index}"
            )
        for row in current.rollouts:
            if row.extras.get("policy_version") != current.policy_version:
                ok = False
                messages.append(
                    f"{row.episode_id}: sampled under {row.extras.get('policy_version')} but landed "
                    f"in round {current.index} ({current.policy_version})"
                )
        if current.table_digest == previous.table_digest:
            ok = False
            messages.append(
                f"round {current.index}: the parameters are unchanged from round {previous.index}, so "
                "the update did nothing"
            )
    if not messages:
        messages.append(
            "versions: " + " -> ".join(result.policy_version for result in results)
            + "; parameter digests all differ"
        )
    return CheckResult(
        name="lineage",
        what_it_checks="each round's rows came from that round's policy, and the update changed it",
        ok=ok,
        messages=messages,
    )


def check_budget(results: Sequence[RoundResult], stage: str) -> CheckResult:
    """The expensive critic ran only where the gate said it would, and every decision was logged."""
    if stage != fusion_policy.FULL:
        return CheckResult(
            name="budget",
            what_it_checks="the expensive critic ran only inside the uncertainty band",
            ok=True,
            messages=["rule-only stage: no expensive critic to budget"],
        )
    messages: list[str] = []
    gates = _all_gates(results)
    ran = sum(1 for gate in gates.values() if " ran:" in gate)
    skipped = sum(1 for gate in gates.values() if " skipped:" in gate)
    missing = [key for key, gate in gates.items() if not gate]
    if missing:
        messages.append(f"{len(missing)} attempt(s) recorded no gate decision, e.g. {missing[0]}")
    messages.append(f"expensive critic ran {ran} time(s), skipped {skipped} time(s)")
    return CheckResult(
        name="budget",
        what_it_checks="the expensive critic ran only inside the uncertainty band",
        ok=not missing,
        messages=messages,
    )


def _all_gates(results: Sequence[RoundResult]) -> dict[str, str]:
    gates: dict[str, str] = {}
    for result in results:
        for episode_id, gate in result.gates.items():
            gates[f"{result.index}:{episode_id}"] = gate
    return gates


# --------------------------------------------------------------------------- the round


def run_round(
    policy: TabularPolicy,
    critic: RuleCritic | BudgetedCritic | FixtureCritic | ModelCritic,
    config: Config,
    round_index: int,
    *,
    rule_critic: RuleCritic | None = None,
) -> RoundResult:
    """Sample every task, judge it, fuse, check, and update the policy once."""
    rule_critic = rule_critic or RuleCritic()
    version = policy.version
    # A served policy has no distribution to measure; see served_policy.py for what replaces it.
    entropy_before = 0.0 if config.served else _mean_entropy(policy, config)
    result = RoundResult(
        index=round_index,
        policy_version=version,
        # A served policy's "parameters" are the checkpoint behind the endpoint; the model id is
        # the only honest digest of that, and it is not pretending to be a table.
        table_digest=(
            f"served:{policy.model}" if config.served else _digest_of_table(policy.table())
        ),
        entropy_before=entropy_before,
    )

    for task_id in config.tasks:
        for sample_index in range(config.samples):
            episode = run_episode(
                policy,
                task_id,
                sample_index=sample_index,
                max_turns=config.max_turns,
                config_hash=config.digest(),
                policy_version=version,
            )
            rule_verdict = rule_critic.critique(episode)
            judged = critic.critique(episode)

            # The critic term exists only when a judge other than the rule produced the verdict;
            # otherwise the stage is rule-only for this attempt and the fusion says so.
            has_judge = judged.source != rule_critic.source
            stage_used = fusion_policy.FULL if has_judge else fusion_policy.RULE_ONLY
            if config.stage == fusion_policy.FULL and not has_judge:
                stage_used = fusion_policy.RULE_ONLY
            terms = fusion_policy.terms_for(
                stage_used,
                rule_outcome=episode.rollout.reward_terms["rule_outcome"],
                rule_turn=progress_reached(episode),
                critic=judged.score if has_judge else None,
            )
            fusion = fusion_policy.fuse_terms(terms, stage_used)

            # The row's terms are the *contributions*, so `rollout.reward()` is the number the
            # trainer consumed and every shared check sees the same arithmetic.  The raw scores
            # stay in extras, so a damped contribution can be traced back.
            episode.rollout.reward_terms = dict(fusion.contributions)
            episode.rollout.extras.update(
                {
                    "stage_used": stage_used,
                    "terms_raw": terms,
                    "fusion_total": round(fusion.total, 6),
                    "critic_source": judged.source,
                    "gate": judged.gate,
                    "conflicts": list(fusion.conflicts)
                    + disagreement(rule_verdict, judged) if has_judge else list(fusion.conflicts),
                }
            )

            result.episodes.append(episode)
            result.verdicts[episode.rollout.episode_id] = judged
            result.rule_verdicts[episode.rollout.episode_id] = rule_verdict
            result.terms_raw[episode.rollout.episode_id] = terms
            result.totals[episode.rollout.episode_id] = fusion.total
            result.conflicts[episode.rollout.episode_id] = list(episode.rollout.extras["conflicts"])
            result.gates[episode.rollout.episode_id] = judged.gate

    # Advantage and update.
    summaries = summarise_groups(result.episodes, result.totals)
    updates = []
    for episode in result.episodes:
        advantage = advantage_for(summaries, episode, result.totals)
        weights = turn_weights(
            episode,
            advantage,
            mode=config.credit,
            turn_scores=_turn_scores_for(episode, result, config),
        )
        from .agent import policy_updates

        updates.extend(policy_updates(episode, weights))
    policy.apply(updates)
    result.updates_applied = len(updates)
    result.entropy_after = 0.0 if config.served else _mean_entropy(policy, config)

    result.checks = run_all(result.rollouts, min_spread_fraction=config.min_spread_fraction)
    if config.served:
        # A frozen policy has no update, so "the parameters changed" is not a failure, it is the
        # mode. The repertoire replaces the entropy reading; see served_policy.py.
        result.checks = [check for check in result.checks if check.name not in {"lineage", "entropy"}]
    return result


def _turn_scores_for(episode: Episode, result: RoundResult, config: Config) -> list[float] | None:
    """Which critic's turn scores shape the weight.  The judge's, when the judge ran."""
    if config.credit != "turn":
        return None
    verdict = result.verdicts[episode.rollout.episode_id]
    return verdict.turn_scores()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def progress_reached(episode: Episode) -> float:
    """How far this attempt got towards the goal: the environment's progress trace, maximised.

    Not the mean of the critic's per-turn scores, and the difference is measured rather than
    stylistic.  With a mean, a policy learns to pad: two neutral turns (0.5, 0.5) followed by a
    wrong report average to 0.67, while the *right* action followed by a wrong report averages to
    0.42 - so padding beats acting, and the first draft of this loop learned exactly that
    (`cat draft.txt` was sampled 37 times in a 24-attempt round).  The maximum asks a different
    and more useful question: how close to the goal did this attempt ever get?  A wrong ending is
    still a wrong ending (the outcome term carries that), but padding no longer pays.
    """
    return max(episode.attempt_progress) if episode.attempt_progress else 0.0


def _mean_entropy(policy: TabularPolicy, config: Config) -> float:
    """Mean entropy over the round's states. Tables only: a served policy has no distribution."""
    """Mean entropy over the states this configuration actually visits.

    Visiting rather than enumerating: a table has a row for every state ever seen, and a row no
    longer reached says nothing about the policy's current behaviour.
    """
    entropies: list[float] = []
    for task_id in config.tasks:
        attempt = env.start(task_id)
        for _ in range(config.max_turns):
            if attempt.finished:
                break
            signature, menu = attempt.signature, tuple(attempt.menu())
            entropies.append(policy.entropy(signature, menu))
            attempt.step(policy.greedy(signature, menu))
    return _mean(entropies)


# --------------------------------------------------------------------------- the loop


def run_loop(
    config: Config,
    *,
    critic: RuleCritic | BudgetedCritic | FixtureCritic | ModelCritic | None = None,
    policy: TabularPolicy | None = None,
    writer: RunWriter | None = None,
    code_sha: str = "git:none",
    runs_dir: str = "runs",
    out: Callable[[str], None] = print,
) -> list[RoundResult]:
    """Run every round, writing one run record for the loop and printing the evidence."""
    critic = critic or RuleCritic()
    policy = policy or TabularPolicy(
        temperature=config.temperature, seed=config.seed, learning_rate=config.learning_rate
    )
    if writer is None:
        header = RunHeader(
            run_id=f"p2-loop-{config.stage}-{config.credit}",
            env_version=env.ENV_VERSION,
            config_hash=config.digest(),
            code_sha=code_sha,
            project=PROJECT,
            dataset="p2 families",
            dataset_digest=config.digest(),
            notes=config.notes(getattr(critic, "source", type(critic).__name__))
            + (f"; policy={policy.version}" if config.served else ""),
        )
        writer = RunWriter(header, root=runs_dir)

    out(
        f"run {writer.header.run_id}: {config.notes(getattr(critic, 'source', type(critic).__name__))}"
        + (f"; policy={policy.version}" if config.served else "")
    )
    results: list[RoundResult] = []
    for round_index in range(config.rounds):
        if not config.served:
            policy.version = f"policy-v{round_index}"
        result = run_round(policy, critic, config, round_index)
        results.append(result)
        writer.append_rollouts(round_index, result.rollouts)
        writer.append_round(round_record(round_index, result.rollouts))
        out(_render_round(result))
        if not config.served:
            policy = policy.bumped(f"policy-v{round_index + 1}")

    final = run_all(results[-1].rollouts, min_spread_fraction=config.min_spread_fraction)
    final.append(check_lineage(results))
    final.append(check_entropy(results))
    final.append(check_budget(results, config.stage))
    for check in final:
        out(check.render())
    out(render_cross_round(results))
    return results


def _render_round(result: RoundResult) -> str:
    terms = ", ".join(f"{name}={value:+.3f}" for name, value in sorted(result.term_means.items()))
    return (
        f"round {result.index} [{result.policy_version}] reward={result.mean_reward:+.3f} "
        f"solved={result.solved_share:.0%} spread={result.spread:.0%} "
        f"entropy={result.entropy_before:.2f}->{result.entropy_after:.2f} "
        f"conflicts={sum(len(items) for items in result.conflicts.values())}  terms: {terms}"
    )


def render_cross_round(results: Sequence[RoundResult], sample_ids: Sequence[str] | None = None) -> str:
    """The pass criterion, printed: per-round means, and specific sample ids re-scored.

    The chapter asks for a log where feedback reaches the next round of *sampling*.  A per-round
    mean cannot show that - the mean moves whether or not any particular attempt changed.  A table
    of the same sample ids, their rewards per round, and the policy version that produced each row
    can: if the ids and their rewards move together with the version, the loop is closed.
    """
    if not results:
        return "no rounds"
    ids = list(sample_ids) if sample_ids else [episode.rollout.episode_id for episode in results[0].episodes[:3]]
    header = "sample id".ljust(22) + "".join(f"r{result.index}".ljust(14) for result in results)
    lines = ["", "the same sample ids, re-scored each round (the pass criterion):", header]
    for episode_id in ids:
        row = episode_id.ljust(22)
        for result in results:
            value = result.totals.get(episode_id)
            row += ("--" if value is None else f"{value:+.3f}").ljust(14)
        lines.append(row)
    # A served policy's version string is longer than a table's ("served:qwen3-1.7b-sft"), and a
    # column that silently overflows runs two rounds together in the evidence printout.
    versions = "versions".ljust(22) + "".join(f"{result.policy_version}".ljust(22) for result in results)
    lines.append(versions)
    return "\n".join(lines)
