"""Build the cold-start demonstrations: the oracle trajectory, with the observation it really produces.

A demonstration has to be a trajectory the runtime would accept, or the fine-tune learns a format
the harness refuses.  So each record is assembled by *running the oracle* in the sandbox and using
its actual stdout as the observation - not a number typed into a file.  If the oracle changes, the
demonstrations change with it, and `--check` fails until they are rebuilt.

Two supervised turns per task, because the runtime asks for two kinds of reply:

    system + question                     ->  <code>oracle</code>
    ... + assistant code + observation    ->  <report>the value the program printed</report>

The second turn is the one that teaches the habit this project needed: read the observation, then
report what it says.  A demonstration that skips it teaches the model to answer from thin air, which
is exactly the failure measured on 2026-11-07 (`p4-orders-biggest-month`: a silent program, then a
guessed "March").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "src"))

from p4_deepanalyze.agent import first_message  # noqa: E402
from p4_deepanalyze.oracles import ORACLE  # noqa: E402
from p4_deepanalyze.protocol import SYSTEM_PROMPT  # noqa: E402
from p4_deepanalyze.sandbox import LocalSandbox  # noqa: E402
from p4_deepanalyze.tasks import load  # noqa: E402

OUT = HERE / "sft.jsonl"


def observation_text(stdout: str) -> str:
    """The exact shape the harness sends back: one value, wrapped in the marker."""
    return f"<observation exit=0>\nstdout:\n{stdout.strip()}\n</observation>"


def build(tasks_path: Path | None = None) -> list[dict]:
    sandbox = LocalSandbox(timeout_s=10)
    records: list[dict] = []
    for task in load(tasks_path or (HERE / "tasks.jsonl")):
        program = ORACLE[task.family]
        observation = sandbox.run(program, task.files())
        if observation.exit_code != 0:
            raise RuntimeError(f"{task.id}: oracle failed: {observation.stderr[-300:]}")
        answer = observation.stdout.strip()
        base = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_message(task)},
        ]
        after_code = base + [
            {"role": "assistant", "content": f"<code>\n{program.strip()}\n</code>"},
            {"role": "user", "content": observation_text(answer)},
        ]
        records.append(
            {
                "task_id": task.id,
                "family": task.family,
                "turns": [
                    {"messages": base, "target": f"<code>\n{program.strip()}\n</code>"},
                    {"messages": after_code, "target": f"<report>{answer}</report>"},
                ],
            }
        )
    return records


def render(records: list[dict]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the P4 cold-start demonstrations")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    records = build()
    text = render(records)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.name} is stale; regenerate with {Path(__file__).name}")
            return 1
        print(f"{OUT.name} is current ({len(text.splitlines())} tasks)")
        return 0
    OUT.write_text(text, encoding="utf-8")
    turns = sum(len(record["turns"]) for record in records)
    print(f"wrote {OUT.name}: {len(records)} tasks, {turns} supervised turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
