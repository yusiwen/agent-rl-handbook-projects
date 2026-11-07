"""Build ``fixtures/eval-claims.json``: every evaluation number Chapter 11 quotes.

The chapter's tables are supposed to be a view of run records, not a second telling of them. This
script reads the records and writes down what they say, so a reader can check the chapter against
one file and the checker can check that file against the records.

    python3 p3-verl/checks/make_eval_claims.py --runs runs --out p3-verl/fixtures/eval-claims.json

Only the fields a claim depends on are copied: which run, which round, which code revision, how
many samples per task, and the solved count per task - plus the solved count of every round, so a
series like 0, 0, 11, 17, 20, 20 is checked as a series rather than as one point. The rollouts stay
in ``runs/``; the manifest names them, it does not replace them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: (run id, round index or -1 for the last, label, section). The label is the chapter's wording; the
#: section is where the number appears. Claiming a training run's *first* round is how an eval of the
#: policy that entered the run is recorded - and it is the number the chapter quotes for a band.
CLAIMS: tuple[tuple[str, int, str, str], ...] = (
    ("p3-rl-served-5", 0, "ten samples per task: eighteen of sixty before the update", "variance"),
    ("p3-rl-served-6", 0, "the calibrated band at temperature 1.0: nothing solved in sixty rollouts", "ceiling"),
    ("p3-rl-served-7", 0, "temperature 1.3 does not move the band", "ceiling"),
    ("p3-rl-served-8", 0, "temperature 1.7 buys four successes in sixty", "ceiling"),
    ("p3-rl-served-9", 0, "the same band after six rounds: four successes become sixty", "ceiling"),
    ("p3-rl-served-10", -1, "the last adapter from that run, sampled greedily: sixty of sixty", "ceiling"),
    ("p3-rl-served-13", 0, "five-step plans at temperature 0: files thirty of thirty, ledger none", "wall"),
    ("p3-rl-served-14", 0, "the same plans at temperature 1.7: files twenty-four, ledger none", "wall"),
    ("p3-rl-served-24", -1, "turn credit: one plan solved at round two, all three gone by round five", "credit"),
    ("p3-rl-served-31", -1, "three-tier credit at the old step size: peaks at fourteen and decays to three", "credit"),
    ("p3-rl-served-37", -1, "three-tier credit at 3e-5: climbs to twenty and stays for two rounds", "credit"),
    ("p3-rl-served-38", -1, "the last adapter from that run, sampled greedily: twenty of thirty", "credit"),
    ("p3-rl-served-39", -1, "the base model on six five-step plans: nothing solved", "demonstrations"),
    ("p3-rl-served-40", -1, "two epochs of demonstrations: forty of sixty", "demonstrations"),
    ("p3-rl-served-41", -1, "four epochs of demonstrations: forty of sixty, composed differently", "demonstrations"),
    ("p3-rl-served-42", -1, "eight epochs of demonstrations: fifty of sixty", "demonstrations"),
    ("p3-rl-served-43", -1, "the reward-trained adapter on the same six plans: fifty of sixty", "demonstrations"),
    ("p2-rl-eval", -1, "held-out tasks before the update: twenty-three of forty", "heldout"),
    ("p2-rl-eval-2", -1, "held-out tasks after the update: thirty of forty, with one task regressed", "heldout"),
)


def rounds_of(run_dir: Path) -> list[int]:
    indices = sorted(int(path.stem.split("-")[1]) for path in run_dir.glob("round-*.jsonl"))
    if not indices:
        raise SystemExit(f"{run_dir}: no round files")
    return indices


def is_solved(row: dict) -> bool:
    """The environment's definition of a solve, not the record's most convenient field.

    ``extras.goal_met`` says the *world* is in the requested state.  ``Attempt.solved()`` says the
    episode finished, the reported answer matches, and the world is right - and the difference is
    not academic: a files task whose directory is correct but that reports 2 where the task asked
    for 1 scores goal_met True and is not solved.  Counting goal_met overstated every table here
    by up to fourteen episodes per run (run 6: 14 vs 0), which is how this predicate was found.
    """

    extras = row.get("extras", {})
    if extras.get("truncated") or not extras.get("goal_met"):
        return False
    reported = extras.get("reported")
    expected = extras.get("expected")
    if reported is None or expected is None:
        return False
    return str(reported).strip().lower() == str(expected).strip().lower()


def solved_by_task(run_dir: Path, round_index: int) -> tuple[dict[str, list[int]], int, int]:
    """Per task: [solved, sampled] for one round, recomputed from the rollouts themselves."""
    path = run_dir / f"round-{round_index:03d}.jsonl"
    counts: dict[str, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("group_id") or row.get("extras", {}).get("task") or "unknown"
        entry = counts.setdefault(task, [0, 0])
        entry[0] += 1 if is_solved(row) else 0
        entry[1] += 1
    solved = sum(entry[0] for entry in counts.values())
    sampled = sum(entry[1] for entry in counts.values())
    return counts, solved, sampled


def build(runs_root: Path) -> dict:
    claims = []
    for run_id, chosen, label, section in CLAIMS:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise SystemExit(f"{run_dir}: missing; this script needs the records, not a summary")
        header = json.loads((run_dir / "header.json").read_text(encoding="utf-8"))
        indices = rounds_of(run_dir)
        index = indices[chosen] if chosen >= 0 else indices[-1]
        counts, solved, sampled = solved_by_task(run_dir, index)
        per_round = [solved_by_task(run_dir, i)[1] for i in indices]
        claims.append(
            {
                "run": run_id,
                "round": index,
                "rounds": len(indices),
                "per_round_solved": per_round,
                "label": label,
                "section": section,
                "code_sha": header["code_sha"],
                "dataset": header.get("dataset", ""),
                "notes": header.get("notes", ""),
                "tasks": {task: counts[task] for task in sorted(counts)},
                "solved": solved,
                "sampled": sampled,
            }
        )
    return {
        "generated_by": "python3 p3-verl/checks/make_eval_claims.py --runs runs",
        "note": (
            "Every number Chapter 11 quotes, recomputed from runs/*/round-NNN.jsonl by the rollouts' "
            "own goal_met field. Regenerate after any new run; eval_claims.py checks the arithmetic "
            "everywhere and the records wherever runs/ exists."
        ),
        "claims": claims,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write the evaluation manifest from run records")
    parser.add_argument("--runs", default="runs", help="directory of run records")
    parser.add_argument("--out", default="p3-verl/fixtures/eval-claims.json")
    args = parser.parse_args(argv)

    payload = build(Path(args.runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    total = sum(claim["sampled"] for claim in payload["claims"])
    print(f"wrote {out}: {len(payload['claims'])} claims, {total} rollouts accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
