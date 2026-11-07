"""Generate and check the numbers Chapter 10 quotes from P4 runs.

Same discipline as Chapter 11's manifest, smaller: the chapter's two headline numbers - 1 of 6
with a slim image, 4 of 6 with one that has pandas - are recomputed from ``episodes.jsonl``
rather than transcribed, and the check refuses a run whose summary disagrees with its rows.

    python3 p4-deepanalyze/checks/p4_claims.py --generate --runs runs
    python3 p4-deepanalyze/checks/p4_claims.py --check              # arithmetic only
    python3 p4-deepanalyze/checks/p4_claims.py --check --runs runs  # recompute too
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "fixtures" / "p4-claims.json"

#: (run id, label, the section of Chapter 10 that cites it)
CLAIMS: tuple[tuple[str, str, str], ...] = (
    ("p4-20261107-190738", "slim image: the programs could not import pandas", "runner"),
    ("p4-20261107-190831", "pandas image: the same policy, the same tasks", "runner"),
)


def recompute(run_dir: Path) -> dict:
    episodes = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    header = json.loads((run_dir / "header.json").read_text(encoding="utf-8"))
    return {
        "run": run_dir.name,
        "code_sha": header["code_sha"],
        "model": episodes[0]["model"] if episodes else header.get("env_version", ""),
        "executor": (episodes[0]["turns"][0]["observation"] or {}).get("executor", "")
        if episodes and episodes[0]["turns"] and episodes[0]["turns"][0].get("observation")
        else "",
        "tasks": len(episodes),
        "solved": sum(1 for e in episodes if e["solved"]),
        "sandbox_errors": sum(e["sandbox_errors"] for e in episodes),
        "timeouts": sum(e["timeouts"] for e in episodes),
        "protocol_error_tasks": sum(1 for e in episodes if e["protocol_errors"]),
        "code_runs": sum(1 for e in episodes for t in e["turns"] if t["kind"] == "code"),
        "per_task": {e["task_id"]: {"solved": e["solved"], "reason": e["reason"]} for e in episodes},
    }


def build(runs_root: Path) -> dict:
    claims = []
    for run_id, label, section in CLAIMS:
        entry = recompute(runs_root / run_id)
        entry.update({"label": label, "section": section})
        claims.append(entry)
    return {"claims": claims}


def problems_of(payload: dict) -> list[str]:
    problems: list[str] = []
    for claim in payload.get("claims", []):
        name = claim.get("run", "?")
        for key in ("run", "code_sha", "model", "tasks", "solved", "sandbox_errors", "code_runs", "per_task", "label", "section"):
            if key not in claim:
                problems.append(f"{name}: missing {key}")
        if claim.get("solved", 0) > claim.get("tasks", 0):
            problems.append(f"{name}: solved {claim['solved']} > tasks {claim['tasks']}")
        if len(claim.get("per_task", {})) != claim.get("tasks", -1):
            problems.append(f"{name}: per_task has {len(claim.get('per_task', {}))} rows for {claim.get('tasks')} tasks")
        unless_solved = sum(1 for row in claim.get("per_task", {}).values() if row.get("solved"))
        if unless_solved != claim.get("solved", -1):
            problems.append(f"{name}: per_task counts {unless_solved} solved, claim says {claim.get('solved')}")
        if claim.get("sandbox_errors", 0) > claim.get("code_runs", 0):
            problems.append(f"{name}: more sandbox errors than code runs")
    return problems


def compare(payload: dict, runs_root: Path) -> list[str]:
    problems: list[str] = []
    for claim in payload.get("claims", []):
        run_dir = runs_root / claim["run"]
        if not run_dir.exists():
            problems.append(f"{claim['run']}: no record at {run_dir}")
            continue
        fresh = recompute(run_dir)
        for key in ("code_sha", "tasks", "solved", "sandbox_errors", "timeouts", "protocol_error_tasks", "code_runs", "per_task"):
            if fresh[key] != claim.get(key):
                problems.append(f"{claim['run']}: {key} is {fresh[key]!r} in the record, {claim.get(key)!r} in the claim")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4 claim manifest")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--runs", default=None)
    args = parser.parse_args(argv)

    if args.generate:
        payload = build(Path(args.runs or "runs"))
        MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(payload['claims'])} claims)")
        return 0
    if not args.check:
        parser.error("pass --generate or --check")

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = problems_of(payload)
    if args.runs:
        problems += compare(payload, Path(args.runs))
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    scope = "arithmetic and records" if args.runs else "arithmetic"
    print(f"p4-claims: {len(payload['claims'])} claims, {scope} agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
