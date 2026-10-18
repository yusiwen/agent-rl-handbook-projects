"""P4: the served judge, inside the budget, on the calibrated band.

    python3 p2-reagent/checks/run_p2_fusion.py \
        --policy-base-url http://192.168.2.40:8001 --policy-model qwen3-1.7b-sft \
        --judge-base-url http://192.168.2.40:8000 --judge-model qwen3-1.7b

Three things are being measured, and they are separable on purpose:

1. **Does the judge hold its format?** One `TURN n SCORE:` block per turn and a `WHY:` line, or it
   raises. A judge that returns 0.0 on a parse failure turns a bug into a reward, so a failure here
   aborts the run with the reply quoted - it is the finding, not an inconvenience.
2. **How often do the rule and the judge disagree?** Only where the budget lets the judge run (the
   uncertain middle), which is the point of the gate: the conflict rate on the whole set would be
   dominated by the episodes the cheap critic already settled.
3. **Is the budget explainable?** Every verdict carries the rule score it decided on, so "the gate
   became a learned surface" is something the log shows.

A pre-flight judges one recorded episode before the sweep, because a 1.7B judge either can or cannot
hold that format on a real trace, and finding out after eight minutes of rollouts is silly. No API
key: the judge is a local vLLM server that authenticates nobody, and `api_key_env=None` is how that
is said rather than an empty key.

The band comes from `fixtures/calibration.json`, which names the run it was read from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))

from p2_reagent import env, fusion_policy  # noqa: E402
from p2_reagent.critics import BudgetedCritic, CriticParseError, ModelCritic, RuleCritic  # noqa: E402
from p2_reagent.loop import Config, run_loop  # noqa: E402
from p2_reagent.served_policy import ServedPolicy  # noqa: E402


def band_tasks(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = tuple(payload["band"])
    if not tasks:
        raise SystemExit(f"no in-band tasks in {path}; run calibrate_tasks.py first")
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4: budgeted served judge on the calibrated band")
    parser.add_argument("--policy-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--policy-model", default="qwen3-1.7b-sft")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--judge-model", default="qwen3-1.7b")
    parser.add_argument("--calibration", default="p2-reagent/fixtures/calibration.json")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--code-sha", default="git:unknown")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    tasks = band_tasks(Path(args.calibration))
    print(f"band: {len(tasks)} task(s) from {args.calibration}")

    judge = ModelCritic(
        base_url=args.judge_base_url,
        model=args.judge_model,
        api_key_env=None,
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    critic = BudgetedCritic(RuleCritic(), judge, low=args.low, high=args.high)

    if not args.skip_preflight:
        # One real attempt through the judge, before the sweep.
        policy = ServedPolicy(args.policy_base_url, args.policy_model, temperature=args.temperature)
        from p2_reagent.agent import run_episode

        probe = run_episode(policy, tasks[0], sample_index=0, policy_version=policy.version)
        try:
            verdict = judge.critique(probe)
        except CriticParseError as exc:
            print(f"PREFLIGHT FAILED on {tasks[0]}: {exc}")
            return 2
        print(
            f"preflight ok: {tasks[0]} judged {len(verdict.per_turn)} turn(s), "
            f"score {verdict.score:.2f}"
        )

    config = Config(
        tasks=tasks,
        rounds=args.rounds,
        samples=args.samples,
        stage=fusion_policy.FULL,
        credit="turn",
        temperature=args.temperature,
        seed=args.seed,
        served=True,
    )
    policy = ServedPolicy(args.policy_base_url, args.policy_model, temperature=args.temperature)
    results = run_loop(config, critic=critic, policy=policy, runs_dir=args.runs_dir, code_sha=args.code_sha)

    gates: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    conflicts: list[str] = []
    for result in results:
        for episode in result.episodes:
            verdict = result.verdicts[episode.rollout.episode_id]
            gates[verdict.gate.split(":")[0]] += 1
            sources[verdict.source] += 1
            conflicts.extend(result.conflicts[episode.rollout.episode_id])
    print()
    print(f"  critic sources: {dict(sources)}")
    print(f"  budget decisions: {dict(gates)}")
    print(f"  rule-vs-judge conflicts: {len(conflicts)}")
    for line in conflicts[:5]:
        print(f"    {line}")
    print()
    print(
        f"FUSION_RESULT policy={args.policy_model} judge={args.judge_model} tasks={len(tasks)} "
        f"rounds={args.rounds} samples={args.samples} "
        f"judged={sources.get('model', 0)} conflicts={len(conflicts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
