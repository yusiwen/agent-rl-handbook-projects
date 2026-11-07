"""Greedy evaluation of a served policy on the P5 world.

    python3 p5-openclaw-rl/env/eval.py --model p5-v7 --base-url http://172.17.0.1:8003

Temperature 0, one episode per task, and a record of *how* each episode failed: a protocol error, a
tool error, a wrong answer, or the turn budget.  Those four numbers separate "the format broke" from
"the policy cannot find the document", which are different problems and, in an asynchronous loop,
different causes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from loop import episode  # noqa: E402 - the loop's own episode function, so eval cannot drift
from p5_openclaw.world import TASKS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="greedy eval of a served P5 policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://172.17.0.1:8003")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = [episode(task.id, args.base_url, args.model, temperature=0.0) for task in TASKS]
    solved = sum(1 for row in rows if row["solved"])
    protocol_errors = sum(row["protocol_errors"] for row in rows)
    tool_errors = sum(row["tool_errors"] for row in rows)
    exhausted = sum(1 for row in rows if not row["solved"] and row["protocol_errors"] == 0 and row["steps"] >= 6)
    for row in rows:
        print(f"  {row['task_id']:18} {'ok ' if row['solved'] else 'no '} steps {row['steps']} "
              f"tools {row['tool_errors']} protocol {row['protocol_errors']} {row['seconds']:.1f}s")
    summary = {
        "model": args.model,
        "tasks": len(rows),
        "solved": solved,
        "protocol_errors": protocol_errors,
        "tool_errors": tool_errors,
        "turn_budget_exhausted": exhausted,
        "steps": sum(row["steps"] for row in rows),
        "seconds": round(sum(row["seconds"] for row in rows), 1),
    }
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "episodes": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
