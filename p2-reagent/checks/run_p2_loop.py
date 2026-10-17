"""Run the P2 loop: rollouts, critics, fusion, checks, an update, and the next round.

    # Offline, no model, no GPU, no network: the rule-only stage on both families.
    python3 p2-reagent/checks/run_p2_loop.py --rounds 6 --samples 8

    # The full stage with the scripted judge, which is how the fusion and the budget are exercised
    # without spending tokens:
    python3 p2-reagent/checks/run_p2_loop.py --stage full --critic fixture \\
        --fixtures p2-reagent/fixtures/critic.jsonl

    # The full stage against a served model (the judge is a second model, Chapter 6's cost):
    OPENAI_API_KEY=... python3 p2-reagent/checks/run_p2_loop.py --stage full --critic model \\
        --base-url http://192.168.2.40:8000 --model qwen3-1.7b

Exit status is 0 when every check passes.  The run record lands in ``runs/`` (gitignored), with
one header for the loop, one line per round, and every rollout row carrying the policy version that
sampled it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p2_reagent import env  # noqa: E402
from p2_reagent.critics import BudgetedCritic, FixtureCritic, ModelCritic, RuleCritic  # noqa: E402
from p2_reagent.fusion_policy import FULL, STAGES  # noqa: E402
from p2_reagent.loop import Config, run_loop  # noqa: E402
from p2_reagent.policy import TabularPolicy  # noqa: E402
from p2_reagent.train import CREDIT_MODES  # noqa: E402

FIXTURES = HERE.parent / "fixtures"


def code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return f"git:{result.stdout.strip()}" if result.returncode == 0 else "git:none"


def build_critic(args: argparse.Namespace):
    """The critic the stage asked for, with the budget wrapped around it when it is expensive."""
    if args.critic != "rule" and args.stage != FULL:
        raise SystemExit(
            f"--critic {args.critic} needs --stage full: the rule-only stage must be able to run with "
            "no judge at all, and silently ignoring a requested judge is how a run's header ends up "
            "describing a pipeline that did not run"
        )
    if args.stage != FULL or args.critic == "rule":
        return RuleCritic()
    if args.critic == "fixture":
        path = Path(args.fixtures) if args.fixtures else FIXTURES / "critic.jsonl"
        if not path.exists():
            raise SystemExit(f"no scripted critique file at {path}")
        expensive = FixtureCritic.from_jsonl(str(path))
    else:
        if not args.base_url or not args.model:
            raise SystemExit("--critic model needs --base-url and --model")
        expensive = ModelCritic(base_url=args.base_url, model=args.model)
    return BudgetedCritic(RuleCritic(), expensive, low=args.band_low, high=args.band_high)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 p2-reagent/checks/run_p2_loop.py",
        description="Chapter 6's loop: process feedback, fusion, GRPO over grouped rollouts.",
    )
    parser.add_argument("--rounds", type=int, default=6, help="how many rounds (default 6)")
    parser.add_argument("--samples", type=int, default=8, help="samples per task, the GRPO group size")
    parser.add_argument("--tasks", nargs="+", help="task ids (default: every task in both families)")
    parser.add_argument("--stage", choices=STAGES, default="rule-only", help="rule-only or full")
    parser.add_argument("--credit", choices=CREDIT_MODES, default="trajectory", help="where a turn's weight comes from")
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--critic", choices=("rule", "fixture", "model"), default="rule")
    parser.add_argument("--fixtures", help="scripted critique file (default fixtures/critic.jsonl)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint for the model judge")
    parser.add_argument("--model", help="judge model name")
    parser.add_argument("--band-low", type=float, default=0.5, help="budget: lower edge of the uncertain band")
    parser.add_argument("--band-high", type=float, default=0.9, help="budget: upper edge")
    parser.add_argument("--runs-dir", default="runs", help="where run records go (gitignored)")
    parser.add_argument("--json", help="also write the round summaries here")
    args = parser.parse_args(argv)

    tasks = tuple(args.tasks) if args.tasks else tuple(task.task_id for task in env.all_tasks())
    config = Config(
        tasks=tasks,
        samples=args.samples,
        rounds=args.rounds,
        stage=args.stage,
        credit=args.credit,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    critic = build_critic(args)
    policy = TabularPolicy(temperature=args.temperature, seed=args.seed, learning_rate=args.learning_rate)
    results = run_loop(config, critic=critic, policy=policy, code_sha=code_sha(), runs_dir=args.runs_dir)

    if args.json:
        payload = [result.summary() for result in results]
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    # Exit status: everything must pass, except that advantage spread is judged on the final round
    # only - a cold policy legitimately has flat groups in round 0, and Chapter 6 makes spread an
    # exit condition of the GRPO stage rather than a precondition.
    failed = [
        check.name
        for result in results
        for check in result.checks
        if not check.ok and check.name != "advantage spread"
    ]
    final = [check for check in results[-1].checks if check.name == "advantage spread"]
    if any(not check.ok for check in final):
        failed.append("advantage spread (final round)")
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
