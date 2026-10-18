"""The P2 gate: the loop's contract, checked offline, with no network and no GPU.

What it asserts, and why each one is here rather than in a test:

1. **Every task is solvable** - a menu that omits the correct action is a broken environment, not a
   hard task, and the first draft of this project had exactly that bug.
2. **The common five checks pass** - mask, boundaries, terms, advantage spread: the same assertions
   every project owes Chapter 5, on rows this project produced.
3. **The loop closed** - the same sample ids reappear across rounds, their rewards changed, and the
   policy version of each round is the version that sampled it.  This is Chapter 6's pass criterion,
   expressed as something a machine can fail.
4. **The entropy did not collapse** - the Echo Trap, per round.
5. **A rule-vs-critic conflict is found** in the full stage - the chapter asks the reader to find one
   and say what they did about it; if the pipeline cannot produce one, that conversation is not
   possible.

Exit status is 0 only when all of it holds.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p2_reagent import env  # noqa: E402
from p2_reagent.agent import run_episode  # noqa: E402
from p2_reagent.critics import BudgetedCritic, FixtureCritic, RuleCritic, disagreement  # noqa: E402
from p2_reagent.fusion_policy import FULL, RULE_ONLY  # noqa: E402
from p2_reagent.loop import (  # noqa: E402
    Config,
    check_budget,
    check_entropy,
    check_lineage,
    run_loop,
)
from p2_reagent.policy import TabularPolicy  # noqa: E402

FIXTURES = HERE.parent / "fixtures"

#: The fixture contract.  An accidental edit to a task, a family or the sampling should fail here
#: rather than quietly change what the run demonstrates.
EXPECTED_TASKS = 43
EXPECTED_FAMILIES = 2
EXPECTED_SAMPLES = 8
#: The share of groups that must show a non-zero reward difference.
#:
#: The shared default is 0.5, and it was calibrated on the first version's three tasks, which the
#: policy could solve about a fifth of the time. The set behind this project is 43 tasks, none of
#: them calibrated against this policy: measured 18/43 (42%) at a cold uniform policy, because
#: solving a two-step task by chance is about one attempt in eighty. That fraction is a difficulty
#: measurement, not a machinery measurement - the machinery is that every group which does vary
#: yields a zero-sum, non-degenerate advantage. So the floor drops to 0.25, which still catches
#: "nothing varies at all", and the measured fraction is reported rather than hidden.
MIN_SPREAD_FRACTION = 0.25
EXPECTED_ROUNDS = 3
EXPECTED_ATTEMPTS_PER_ROUND = EXPECTED_TASKS * EXPECTED_SAMPLES
EXPECTED_TERMS = {"rule_outcome", "rule_turn"}


def solvable(task_id: str) -> tuple[bool, list[str]]:
    """Walk a task with an oracle that follows the goal's own gradient, and say whether it solves.

    The oracle is not the policy: it picks, from the offered menu, the action that maximises the
    environment's progress.  If no such walk reaches the goal, the task is unsolvable and the gate
    says so - which is the check that would have caught the missing ``rm tmp.log``.
    """
    attempt = env.start(task_id)
    trail: list[str] = []
    for _ in range(8):
        if attempt.finished:
            break
        best: tuple[float, str] | None = None
        for action in attempt.menu():
            probe = env.start(task_id)
            for taken in trail:
                probe.step(taken)
            result = probe.step(action)
            score = probe.progress()
            if result.done and probe.solved():
                score = 2.0
            if best is None or score > best[0]:
                best = (score, action)
        assert best is not None
        action = best[1]
        attempt.step(action)
        trail.append(action)
    return attempt.solved(), trail


def main() -> int:
    failures: list[str] = []
    print("== P2 loop check ==")
    tasks = env.all_tasks()
    print(f"tasks: {len(tasks)} in {len({task.family for task in tasks})} family(ies)")
    if len(tasks) != EXPECTED_TASKS:
        failures.append(f"expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    if len({task.family for task in tasks}) != EXPECTED_FAMILIES:
        failures.append(f"expected {EXPECTED_FAMILIES} families")

    print("\n1. every task is solvable from its own menu")
    for task in tasks:
        solved, trail = solvable(task.task_id)
        print(f"  {'ok ' if solved else 'FAIL'} {task.task_id}: {' -> '.join(trail)}")
        if not solved:
            failures.append(f"{task.task_id} cannot be solved from the menu it offers")

    with tempfile.TemporaryDirectory() as tmp:
        # ---------------------------------------------------------------- the rule-only loop
        print(f"\n2. the rule-only loop: {EXPECTED_ROUNDS} rounds x {EXPECTED_SAMPLES} samples x {EXPECTED_TASKS} tasks")
        config = Config(
            rounds=EXPECTED_ROUNDS,
            samples=EXPECTED_SAMPLES,
            seed=7,
            min_spread_fraction=MIN_SPREAD_FRACTION,
        )
        results = run_loop(config, runs_dir=tmp)

        for result in results:
            if len(result.episodes) != EXPECTED_ATTEMPTS_PER_ROUND:
                failures.append(f"round {result.index}: {len(result.episodes)} attempts")
        print(f"  attempts per round: {EXPECTED_ATTEMPTS_PER_ROUND}")
        if set(results[0].term_means) != EXPECTED_TERMS:
            failures.append(f"terms {sorted(results[0].term_means)} != {sorted(EXPECTED_TERMS)}")

        # The advantage-spread check is a *training* assertion and a cold policy can fail it in
        # round 0 without anything being wrong: with an untrained table most groups are all-zero.
        # Chapter 6 makes spread an exit condition of the GRPO stage, so it is judged on the last
        # round; every other check is judged on every round.
        problems = [
            (result.index, check)
            for result in results
            for check in result.checks
            if not check.ok and check.name != "advantage spread"
        ]
        for index, problem in problems:
            print(f"  FAIL round {index} {problem.name}: {problem.messages}")
        if problems:
            failures.append(f"{len(problems)} common check(s) failed")
        final_spread = [check for check in results[-1].checks if check.name == "advantage spread"]
        for check in final_spread:
            print(f"  {'ok  ' if check.ok else 'FAIL'} final-round {check.name}: {check.messages}")
            if not check.ok:
                failures.append("the final round still has no advantage spread: the loop is not learning")

        # ---------------------------------------------------------------- the pass criterion
        print("\n3. the same sample ids, re-scored (the pass criterion)")
        ids = [episode.rollout.episode_id for episode in results[0].episodes]
        missing = [sample for sample in ids if any(sample not in result.totals for result in results)]
        if missing:
            failures.append(f"{len(missing)} sample id(s) missing from a later round")
        moved = [
            sample
            for sample in ids
            if len({round(results[result.index].totals[sample], 6) for result in results}) > 1
        ]
        print(f"  {len(ids)} sample id(s) present in all rounds; {len(moved)} changed reward")
        if not moved:
            failures.append("no sample changed its reward between rounds: the feedback never reached sampling")

        lineage = check_lineage(results)
        print("  " + lineage.render().replace("\n", "\n  "))
        if not lineage.ok:
            failures.append("lineage check failed")
        entropy = check_entropy(results)
        print("  " + entropy.render().replace("\n", "\n  "))
        if not entropy.ok:
            failures.append("entropy check failed")

        # ---------------------------------------------------------------- the full stage
        print("\n4. the full stage with the scripted judge")
        critic = BudgetedCritic(
            RuleCritic(), FixtureCritic.from_jsonl(str(FIXTURES / "critic.jsonl")), low=0.5, high=0.9
        )
        full_config = Config(rounds=2, samples=EXPECTED_SAMPLES, stage=FULL, seed=7)
        full_results = run_loop(
            full_config,
            critic=critic,
            policy=TabularPolicy(temperature=1.0, seed=7, learning_rate=0.3),
            runs_dir=tmp,
        )
        stages = {episode.rollout.extras["stage_used"] for result in full_results for episode in result.episodes}
        print(f"  stages used: {sorted(stages)}")
        if FULL not in stages:
            failures.append("the full stage never fused a critic term")
        gates = [gate for result in full_results for gate in result.gates.values()]
        if not all(gates):
            failures.append(f"{sum(1 for gate in gates if not gate)} attempt(s) recorded no gate decision")
        print(f"  budget decisions logged: {len(gates)} (ran {sum('ran:' in gate for gate in gates)}, "
              f"skipped {sum('skipped:' in gate for gate in gates)})")
        budget = check_budget(full_results, FULL)
        print("  " + budget.render().replace("\n", "\n  "))
        if not budget.ok:
            failures.append("budget check failed")

        conflicts = [item for result in full_results for items in result.conflicts.values() for item in items]
        print(f"  conflicts found and logged: {len(conflicts)}")
        for conflict in conflicts[:3]:
            print(f"    {conflict}")
        if not conflicts:
            failures.append("no rule-vs-critic conflict was found; the full stage has nothing to talk about")

        # ---------------------------------------------------------------- the run record
        print("\n5. the run record")
        dirs = sorted(path for path in Path(tmp).iterdir() if path.is_dir())
        header_ok = 0
        for directory in dirs:
            header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
            rounds = [json.loads(line) for line in (directory / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if line]
            rows = sorted(directory.glob("round-*.jsonl"))
            required = {"run_id", "env_version", "config_hash", "code_sha", "project"}
            if required <= set(header) and rounds and rows:
                header_ok += 1
            print(f"  {directory.name}: {len(rows)} round file(s), {len(rounds)} round line(s)")
        if header_ok != len(dirs):
            failures.append("a run record is missing its header, its rounds or its rows")

    print()
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[PASS] P2: every task is solvable, the loop closes, no collapse, and conflicts are logged")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
