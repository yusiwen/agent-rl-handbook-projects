"""Tasks: a table, a question, and an answer the harness can check without a judge.

The tasks are deliberately small and local. ``DS-1000`` and the table-question corpora in the
README are the real data for this project, but a first end-to-end run does not need a
download to be honest: six tables whose answers are arithmetic are enough to measure whether
the loop works - and they can be committed, which the share-alike corpora cannot.

A task's ``expected`` is computed by the fixture author (see ``fixtures/build_tasks.py``,
which recomputes every answer from the CSV and refuses to write a file where the two
disagree). It is a number, so the comparison in ``verify`` is exact up to a tolerance rather
than a judgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    table: str  # the CSV itself, one header row and at least one data row
    expected: str
    tolerance: float = 1e-6

    def files(self) -> dict[str, str]:
        return {"table.csv": self.table if self.table.endswith("\n") else self.table + "\n"}

    def preview(self, rows: int = 5) -> str:
        """The header plus a few rows: enough to write code against, not enough to answer."""
        lines = self.table.strip().splitlines()
        head = lines[: 1 + rows]
        suffix = "" if len(lines) <= len(head) else f"\n... ({len(lines) - len(head)} more rows)"
        return "\n".join(head) + suffix

    def columns(self) -> list[str]:
        return [part.strip() for part in self.table.strip().splitlines()[0].split(",")]


def load(path: str | Path) -> list[Task]:
    """Read tasks, and refuse a file that could produce an ambiguous result."""
    tasks: list[Task] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        raw = json.loads(line)
        task = Task(
            id=str(raw["id"]),
            question=str(raw["question"]),
            table=str(raw["table"]),
            expected=str(raw["expected"]),
            tolerance=float(raw.get("tolerance", 1e-6)),
        )
        if task.id in seen:
            raise ValueError(f"{path}:{number}: duplicate task id {task.id!r}")
        if not task.table.strip():
            raise ValueError(f"{path}:{number}: task {task.id} has an empty table")
        if len(task.table.strip().splitlines()) < 2:
            raise ValueError(f"{path}:{number}: task {task.id} needs a header and at least one row")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: no tasks")
    return tasks
