"""Stage 1: the cold start, on traces the world itself produces.

    python3 p5-openclaw-rl/env/sft.py --out /models/p5-sft

The demonstrations are the oracle episodes - open the documents the task names, compute if the task
needs arithmetic, submit - rendered into the exact message shape the loop uses.  Nothing is written
by hand, so the traces cannot describe a tool call the world would refuse, and a change to the world
regenerates them.

Two things this stage buys, and one it does not:

* the **format**: one JSON object per reply, or the protocol refuses it.  A base model asked for this
  format usually complies for a turn or two and then drifts into prose;
* the **habit**: `submit` at the end, with the value the question asked for rather than a summary of
  the documents;
* it does **not** teach tool *selection* under uncertainty - six traces do not contain a search
  strategy.  That is what stage 2 is for, and the record says so when the numbers do not move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))

from p5_openclaw import protocol  # noqa: E402
from p5_openclaw.world import TASKS, World, oracle_episode  # noqa: E402

OUT = HERE.parent / "fixtures" / "demonstrations.jsonl"


def trace(task) -> list[dict]:
    """Replay the oracle, capturing (messages before, the reply to supervise) for each step."""
    world = World(task=task)
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": protocol.first_message(task.question)},
    ]
    turns: list[dict] = []
    for doc in task.oracle_docs:
        reply = json.dumps({"tool": "open", "args": {"doc": doc}}, separators=(", ", ": "))
        turns.append({"messages": list(messages), "reply": reply})
        observation = world.step("open", {"doc": doc})
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": protocol.format_observation(observation.text)})
    if task.oracle_expression:
        reply = json.dumps({"tool": "calc", "args": {"expression": task.oracle_expression}}, separators=(", ", ": "))
        turns.append({"messages": list(messages), "reply": reply})
        observation = world.step("calc", {"expression": task.oracle_expression})
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": protocol.format_observation(observation.text)})
    reply = json.dumps({"tool": "submit", "args": {"answer": task.answer}}, separators=(", ", ": "))
    turns.append({"messages": list(messages), "reply": reply})
    world.step("submit", {"answer": task.answer})
    return [{"task_id": task.id, "messages": turn["messages"], "reply": turn["reply"]} for turn in turns]


def build() -> list[dict]:
    rows: list[dict] = []
    for task in TASKS:
        world = oracle_episode(task)
        if world.submitted is None:
            raise RuntimeError(f"{task.id}: the oracle did not submit")
        rows.extend(trace(task))
    return rows


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the P5 demonstrations")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rows = build()
    text = render(rows)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.name} is stale; regenerate")
            return 1
        print(f"{OUT.name} is current ({len(rows)} turns)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(rows)} supervised turns over {len(TASKS)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
