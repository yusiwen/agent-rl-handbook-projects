"""Check ``fixtures/eval-claims.json``: the arithmetic everywhere, the provenance where it exists.

Chapter 11's tables are generated from run records and committed as one manifest. Two things can go
wrong with such a file, and they need different checks:

* **The arithmetic inside it.** Per-task counts must add up to the claim's total, every task must
  have the same number of samples, and a per-round series must end where the cited round says it
  does. This holds on any machine, so the gate runs it on every machine.
* **Its relationship to the records.** ``runs/`` is gitignored, so on a clone there is nothing to
  compare against. Where the directory exists, ``--runs`` recomputes every claim from the rollouts
  and refuses a mismatch - including the code revision, which is what makes a number ownable.

    python3 p3-verl/checks/eval_claims.py --check
    python3 p3-verl/checks/eval_claims.py --check --runs runs
    python3 p3-verl/checks/eval_claims.py --selftest

The selftest exists because the first version of the generator counted ``extras.goal_met``, which
says the *world* is right, not that the answer was reported correctly: it overstated every table by
up to fourteen episodes per run. A check that shares that mistake is worthless, so the selftest
builds a rollout whose world is right and whose report is wrong, and requires it to count as zero.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_eval_claims  # noqa: E402  (same directory, loaded by path)

SHA = re.compile(r"^git:[0-9a-f]{7,40}$")
REQUIRED = ("run", "round", "rounds", "per_round_solved", "label", "section", "code_sha", "tasks", "solved", "sampled")


def problems_of(payload: dict) -> list[str]:
    """Every internal inconsistency, as text, so a failure names the claim rather than a line."""
    problems: list[str] = []
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["the manifest has no claims"]

    seen: set[tuple[str, int]] = set()
    for index, claim in enumerate(claims):
        missing = [key for key in REQUIRED if key not in claim]
        if missing:
            problems.append(f"claim {index}: missing {', '.join(missing)}")
            continue
        name = f"{claim['run']} round {claim['round']}"
        if not SHA.match(str(claim["code_sha"])):
            problems.append(f"{name}: code_sha {claim['code_sha']!r} is not git:<sha>")
        if claim["section"] not in {"variance", "ceiling", "wall", "credit", "demonstrations", "heldout"}:
            problems.append(f"{name}: unknown section {claim['section']!r}")
        key = (claim["run"], claim["round"])
        if key in seen:
            problems.append(f"{name}: claimed twice")
        seen.add(key)

        tasks = claim["tasks"]
        if not tasks:
            problems.append(f"{name}: no tasks")
            continue
        per_task_solved = 0
        per_task_sampled = 0
        denominators = set()
        for task, counts in tasks.items():
            if not isinstance(counts, list) or len(counts) != 2:
                problems.append(f"{name}: {task} is not [solved, sampled]")
                continue
            solved, sampled = counts
            if sampled <= 0 or solved < 0 or solved > sampled:
                problems.append(f"{name}: {task} counts {counts} are impossible")
            per_task_solved += solved
            per_task_sampled += sampled
            denominators.add(sampled)
        if len(denominators) != 1:
            problems.append(f"{name}: tasks have different sample counts {sorted(denominators)}")
        if per_task_solved != claim["solved"]:
            problems.append(f"{name}: tasks sum to {per_task_solved} solved, claim says {claim['solved']}")
        if per_task_sampled != claim["sampled"]:
            problems.append(f"{name}: tasks sum to {per_task_sampled} sampled, claim says {claim['sampled']}")

        series = claim["per_round_solved"]
        if len(series) != claim["rounds"]:
            problems.append(f"{name}: {len(series)} entries for {claim['rounds']} rounds")
        if series and max(series) > claim["sampled"]:
            problems.append(f"{name}: a round solves more than it sampled")
        if series and 0 <= claim["round"] < len(series) and series[claim["round"]] != claim["solved"]:
            problems.append(
                f"{name}: round {claim['round']} of the series is {series[claim['round']]}, claim says "
                f"{claim['solved']}"
            )
    return problems


def compare_with_runs(payload: dict, runs_root: Path) -> list[str]:
    """Recompute every claim from the rollouts it names, and refuse any difference."""
    problems: list[str] = []
    for claim in payload["claims"]:
        run_dir = runs_root / claim["run"]
        if not run_dir.exists():
            problems.append(f"{claim['run']}: no record at {run_dir}")
            continue
        header = json.loads((run_dir / "header.json").read_text(encoding="utf-8"))
        if header["code_sha"] != claim["code_sha"]:
            problems.append(
                f"{claim['run']}: header says {header['code_sha']}, claim says {claim['code_sha']}"
            )
        indices = make_eval_claims.rounds_of(run_dir)
        series = [make_eval_claims.solved_by_task(run_dir, i)[1] for i in indices]
        if series != claim["per_round_solved"]:
            problems.append(f"{claim['run']}: series {series} != claim {claim['per_round_solved']}")
        if len(indices) != claim["rounds"]:
            problems.append(f"{claim['run']}: {len(indices)} rounds recorded, claim says {claim['rounds']}")
        if claim["round"] not in indices:
            problems.append(f"{claim['run']}: no round {claim['round']}")
            continue
        counts, solved, sampled = make_eval_claims.solved_by_task(run_dir, claim["round"])
        if counts != claim["tasks"] or solved != claim["solved"] or sampled != claim["sampled"]:
            problems.append(
                f"{claim['run']}: round {claim['round']} recomputes to {solved}/{sampled}, claim says "
                f"{claim['solved']}/{claim['sampled']}"
            )
    return problems


def selftest() -> int:
    """A world-right, answer-wrong rollout must not count as solved; a drifting manifest must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        run_dir = runs / "t-run"
        run_dir.mkdir(parents=True)
        rows = [
            # world right, report wrong: goal_met True, not solved
            {"group_id": "t-a", "extras": {"goal_met": True, "reported": "2", "expected": "1", "truncated": False}},
            # world right, report right
            {"group_id": "t-a", "extras": {"goal_met": True, "reported": "1", "expected": "1", "truncated": False}},
            # never finished
            {"group_id": "t-a", "extras": {"goal_met": False, "reported": None, "expected": "1", "truncated": True}},
            {"group_id": "t-b", "extras": {"goal_met": True, "reported": "4", "expected": "4", "truncated": False}},
            {"group_id": "t-b", "extras": {"goal_met": True, "reported": "4", "expected": "4", "truncated": False}},
            {"group_id": "t-b", "extras": {"goal_met": True, "reported": "5", "expected": "4", "truncated": False}},
        ]
        (run_dir / "round-000.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        (run_dir / "header.json").write_text(
            json.dumps({"code_sha": "git:abcdef1", "dataset": "t", "notes": "selftest"}),
            encoding="utf-8",
        )

        original = make_eval_claims.CLAIMS
        make_eval_claims.CLAIMS = (("t-run", -1, "selftest claim", "variance"),)
        try:
            payload = make_eval_claims.build(runs)
        finally:
            make_eval_claims.CLAIMS = original

        claim = payload["claims"][0]
        if claim["solved"] != 3 or claim["sampled"] != 6:
            print(f"selftest: expected 3/6 solved, got {claim['solved']}/{claim['sampled']}", file=sys.stderr)
            return 1
        if problems_of(payload):
            print(f"selftest: fresh manifest has problems: {problems_of(payload)}", file=sys.stderr)
            return 1
        if compare_with_runs(payload, runs):
            print(f"selftest: fresh manifest disagrees with its records: {compare_with_runs(payload, runs)}", file=sys.stderr)
            return 1

        # Now the three ways a manifest can lie, each of which must be caught.
        mutations = {
            "inflated total": lambda p: p["claims"][0].__setitem__("solved", p["claims"][0]["solved"] + 1),
            "wrong series": lambda p: p["claims"][0]["per_round_solved"].__setitem__(0, 4),
            "invented revision": lambda p: p["claims"][0].__setitem__("code_sha", "git:deadbee"),
        }
        for label, mutate in mutations.items():
            broken = json.loads(json.dumps(payload))
            mutate(broken)
            internal = problems_of(broken)
            against_runs = compare_with_runs(broken, runs)
            if not internal and not against_runs:
                print(f"selftest: {label} was accepted", file=sys.stderr)
                return 1
    print("selftest: goal_met is not solved, and three kinds of lie are rejected")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check the evaluation manifest")
    parser.add_argument("--manifest", default=str(HERE.parent / "fixtures" / "eval-claims.json"))
    parser.add_argument("--check", action="store_true", help="check the arithmetic (always possible)")
    parser.add_argument("--runs", help="also recompute from this run-record directory")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.check:
        parser.error("nothing to do: pass --check, --runs or --selftest")

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    problems = problems_of(payload)
    if args.runs:
        problems += compare_with_runs(payload, Path(args.runs))
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"eval-claims: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    scope = "arithmetic and records" if args.runs else "arithmetic"
    print(f"eval-claims: {len(payload['claims'])} claims, {scope} agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
