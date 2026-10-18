"""Which tasks sit in the band, read from a recorded run rather than guessed.

    python3 p2-reagent/checks/calibrate_tasks.py --run runs/p2-loop-rule-only-turn-6
    python3 p2-reagent/checks/calibrate_tasks.py --run runs/p2-loop-rule-only-turn-6 --out fixtures/calibration.json

Why this exists: the served loop failed the advantage-spread check at 14% of groups, because 72% of
episodes were solved and most groups were three-for-three. A cold policy is flat because no sample
succeeds; a competent one is flat because they all do. Neither produces a gradient, and the fix is
not in the loop - it is in which tasks the loop is run on.

So the band is read off the data: per task, how many of the recorded attempts were solved. Tasks at
the extremes are dropped, tasks near the middle are kept, and the selection is written to a file with
the run it came from, so a later experiment can say which calibration it used. Recomputing the band
from a different run is a different calibration, and the file should say so.

The bands are thirds by default because three samples per task is what the served loop collected:
a rate can only be 0, 1/3, 2/3 or 1, and "near half" means the two middle values.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))

from common.schema import read_rollouts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="read the in-band tasks out of a recorded run")
    parser.add_argument("--run", default="runs/p2-loop-rule-only-turn-6")
    parser.add_argument("--out", default="p2-reagent/fixtures/calibration.json")
    parser.add_argument("--low", type=float, default=1 / 3, help="lowest solved share to keep")
    parser.add_argument("--high", type=float, default=2 / 3, help="highest solved share to keep")
    args = parser.parse_args(argv)

    run = Path(args.run)
    rows = []
    for path in sorted(run.glob("round-*.jsonl")):
        rows.extend(read_rollouts(str(path)))
    if not rows:
        print(f"no rollouts in {run}")
        return 2

    attempts: dict[str, int] = defaultdict(int)
    solved: dict[str, int] = defaultdict(int)
    for row in rows:
        task = row.extras.get("task")
        attempts[task] += 1
        solved[task] += 1 if row.reward_terms.get("rule_outcome", 0.0) > 0 else 0

    rates = {task: solved[task] / attempts[task] for task in attempts}
    band = sorted(task for task, rate in rates.items() if args.low <= rate <= args.high)
    never = sorted(task for task, rate in rates.items() if rate == 0.0)
    always = sorted(task for task, rate in rates.items() if rate == 1.0)

    print(f"{len(rates)} tasks, {len(rows)} attempts from {run}")
    print()
    for task in sorted(rates):
        value = rates[task]
        mark = "band" if task in band else ("never" if value == 0.0 else ("always" if value == 1.0 else "out"))
        print(f"  {task:22s} {solved[task]:3d}/{attempts[task]:<3d} {rates[task]:5.0%}  {mark}")
    print()
    print(f"  band ({args.low:.2f}-{args.high:.2f}): {len(band)} tasks")
    print(f"  never solved: {len(never)}   always solved: {len(always)}")

    payload = {
        "run": str(run),
        "attempts": len(rows),
        "low": args.low,
        "high": args.high,
        "band": band,
        "never": never,
        "always": always,
        "rates": {task: round(rate, 4) for task, rate in sorted(rates.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print(f"wrote {out}")
    print("CALIBRATION band=" + ",".join(band))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
