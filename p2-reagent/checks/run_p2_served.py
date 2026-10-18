"""Run the loop with the fine-tuned checkpoint as the policy.

    python3 p2-reagent/checks/run_p2_served.py --base-url http://192.168.2.40:8001 --model qwen3-1.7b-sft

What this stage is for, and what it is not. The policy is a remote endpoint, so nothing is trained
here: the weights the credit machinery computes have nowhere to go. That is deliberate - the
question this run answers is whether the machinery produces a sensible credit signal on traces from
a *competent* policy, rather than on a table that had to be bootstrapped first.

So the reading is expected to be flat across rounds, and a flat reading is the correct result. What
would be wrong is to report it as learning, or to keep the lineage and entropy checks and let them
fail for a reason that is the experiment's design. `Config.served` drops those two checks with the
reason written at the call site, and this runner reports the repertoire - the distinct-action ratio
- in their place.

The same sample ids reappear every round, which is what makes the rounds comparable: a reward that
moves between rounds moved because the critic saw different traces, not because the task changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))

from p2_reagent import env, fusion_policy  # noqa: E402
from p2_reagent.loop import Config, run_loop  # noqa: E402
from p2_reagent.served_policy import ServedPolicy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the loop against a served policy")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3-1.7b-sft")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--code-sha", default="git:unknown")
    args = parser.parse_args(argv)

    tasks = tuple(args.tasks) if args.tasks else tuple(task.task_id for task in env.all_tasks())
    config = Config(
        tasks=tasks,
        rounds=args.rounds,
        samples=args.samples,
        stage=fusion_policy.RULE_ONLY,
        credit="turn",
        temperature=args.temperature,
        seed=args.seed,
        max_turns=args.max_turns,
        served=True,
    )
    policy = ServedPolicy(args.base_url, args.model, temperature=args.temperature)

    results = run_loop(config, policy=policy, runs_dir=args.runs_dir, code_sha=args.code_sha)
    print()
    for result in results:
        stats = policy.summary()
        solved = sum(1 for episode in result.episodes if episode.solved)
        print(
            f"round {result.index} [served] solved {solved}/{len(result.episodes)} "
            f"({solved / len(result.episodes):.0%}) mean reward {sum(result.totals.values()) / len(result.totals):+.3f} "
            f"distinct_actions {stats['distinct_actions']} ratio {stats['distinct_action_ratio']:.3f} "
            f"calls {stats['calls']} reasons {stats['reasons']}"
        )
    stats = policy.summary()
    solved_by_round = [
        sum(1 for episode in result.episodes if episode.solved) / len(result.episodes) for result in results
    ]
    print()
    print(
        f"SERVED_LOOP model={args.model} rounds={args.rounds} samples={args.samples} tasks={len(tasks)} "
        f"solved_by_round={[round(value, 3) for value in solved_by_round]} "
        f"distinct_action_ratio={stats['distinct_action_ratio']:.3f} calls={stats['calls']} "
        f"seconds={stats['seconds']} unparsed={stats['reasons'].get('no_action_line', 0)}"
    )
    print(json.dumps({"policy": stats, "solved_by_round": solved_by_round}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
