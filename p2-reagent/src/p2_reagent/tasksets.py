"""Task specifications, as data rather than as branches.

A task set of forty tasks written as forty ``if task_id == ...`` branches is not a task set, it is
forty bugs waiting for a rename. The specification says what the world starts as, what counts as
done, and what the answer should be; the family code says only how the world responds to an action.
Adding a task is then a line of data, and the properties worth checking become mechanical: every
task solvable, every answer recomputable, two attempts identical, the menu containing the right
action.

The file ``fixtures/tasks.jsonl`` is generated with a fixed seed by ``checks/make_tasks.py`` and
committed. Committing it keeps the set frozen - a change is a visible diff - while the generator
keeps it reproducible.

Two families, because two is enough to show that the machinery is not tied to one world:

* ``files`` - rename or delete one file, then report a count that depends on the final directory.
* ``ledger`` - apply an ordered sequence of movements to a balance, then report it.

Both need the model to act *and* to count, which is deliberate: an agent that reports the right
number after the wrong action has not solved the task, and a task whose answer can be guessed
without acting teaches nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    """One task: the world, the goal, and the answer the environment will accept."""

    task_id: str
    family: str
    op: str
    world: tuple[str, ...]
    expected: str
    source: str = ""
    target: str = ""
    count_mode: str = ""
    count_text: str = ""
    plan: tuple[int, ...] = ()
    distractors: tuple[str, ...] = ()

    # ------------------------------------------------------------------ construction
    def as_dict(self) -> dict:
        payload: dict = {
            "task_id": self.task_id,
            "family": self.family,
            "op": self.op,
            "world": list(self.world),
            "expected": self.expected,
        }
        if self.source:
            payload["source"] = self.source
        if self.target:
            payload["target"] = self.target
        if self.count_mode:
            payload["count"] = {"mode": self.count_mode}
            if self.count_text:
                payload["count"]["text"] = self.count_text
        if self.plan:
            payload["plan"] = list(self.plan)
        if self.distractors:
            payload["distractors"] = list(self.distractors)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskSpec":
        count = payload.get("count") or {}
        return cls(
            task_id=payload["task_id"],
            family=payload["family"],
            op=payload["op"],
            world=tuple(payload["world"]),
            expected=str(payload["expected"]),
            source=payload.get("source", ""),
            target=payload.get("target", ""),
            count_mode=count.get("mode", ""),
            count_text=count.get("text", ""),
            plan=tuple(payload.get("plan", ())),
            distractors=tuple(payload.get("distractors", ())),
        )

    # ------------------------------------------------------------------ what the task says
    @property
    def prompt(self) -> str:
        if self.family == "files":
            verb = (
                f"rename {self.source} to {self.target}"
                if self.op == "rename"
                else f"delete {self.source}"
            )
            ask = (
                f"how many files start with the letter {self.count_text}"
                if self.count_mode == "prefix"
                else "how many files remain"
            )
            return f"In the shared directory, {verb}, then report {ask}."
        if self.family == "ledger":
            steps = []
            for move in self.plan:
                steps.append(f"add {move}" if move > 0 else f"subtract {-move}")
            return (
                f"Starting from a balance of 0, {' then '.join(steps)}, "
                "then report the balance."
            )
        raise ValueError(f"no prompt for family {self.family!r}")

    @property
    def opening(self) -> str:
        if self.family == "files":
            return "directory: " + ", ".join(sorted(self.world))
        return "balance: 0"

    # ------------------------------------------------------------------ what the world offers
    @property
    def correct_action(self) -> str:
        if self.family == "files":
            if self.op == "rename":
                return f"mv {self.source} {self.target}"
            return f"rm {self.source}"
        raise ValueError(f"no single correct action for family {self.family!r}")

    def menu(self) -> tuple[str, ...]:
        """Every legal action, the correct one and the distractors alike.

        The menu is derived from the specification rather than written beside it, because the
        first draft of this project shipped a shared menu that omitted the delete task's own
        action - the task was unsolvable and nothing in the code said so. Deriving it makes that
        class of bug a failed assertion instead of a silent wall.
        """
        expected = int(self.expected)
        entries: set[str] = {"ls" if self.family == "files" else ""}
        entries.discard("")
        entries.add("done 0")
        entries.add(f"done {expected}")
        if expected > 0:
            entries.add(f"done {expected - 1}")
        entries.add(f"done {expected + 1}")

        if self.family == "files":
            entries.add(self.correct_action)
            entries.update(f"cat {name}" for name in self.world)
            for name in self.world:
                if name != self.source:
                    entries.add(f"rm {name}")
                    break
        elif self.family == "ledger":
            for move in self.plan:
                entries.add(f"add {move}" if move > 0 else f"sub {-move}")
            entries.update(self.distractors)
        else:
            raise ValueError(f"no menu for family {self.family!r}")
        return tuple(sorted(entries))

    # ------------------------------------------------------------------ what counts as solved
    def final_files(self) -> tuple[str, ...]:
        """The directory after the task's own action, which is what the answer counts."""
        if self.family != "files":
            raise ValueError("only the files family has a directory")
        remaining = [name for name in self.world if name != self.source]
        if self.op == "rename":
            remaining.append(self.target)
        return tuple(sorted(remaining))

    def recomputed_answer(self) -> str:
        """The answer the specification implies. ``expected`` must equal this."""
        if self.family == "files":
            if self.count_mode == "prefix":
                return str(sum(1 for name in self.final_files() if name.startswith(self.count_text)))
            return str(len(self.final_files()))
        return str(sum(self.plan))

    def contradiction(self) -> str | None:
        """A reason this specification is internally inconsistent, or ``None``."""
        if self.family == "files":
            if self.source not in self.world:
                return "source is not in the world"
            if self.op == "rename" and (not self.target or self.target in self.world):
                return "rename needs a target that is not already present"
            if self.op not in {"rename", "delete"}:
                return f"unknown op {self.op!r}"
        elif self.family == "ledger":
            if not self.plan:
                return "a ledger task needs a plan"
            if any(move == 0 for move in self.plan):
                return "a zero move is not a move"
        else:
            return f"unknown family {self.family!r}"
        if self.expected != self.recomputed_answer():
            return f"expected {self.expected!r} but the specification implies {self.recomputed_answer()!r}"
        return None


def load(path: Path) -> tuple[TaskSpec, ...]:
    """Read a task set, refusing a set that contradicts itself."""
    specs = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        spec = TaskSpec.from_dict(json.loads(line))
        if spec.task_id in seen:
            raise ValueError(f"duplicate task id {spec.task_id!r}")
        seen.add(spec.task_id)
        problem = spec.contradiction()
        if problem is not None:
            raise ValueError(f"{spec.task_id}: {problem}")
        specs.append(spec)
    if not specs:
        raise ValueError(f"no tasks in {path}")
    return tuple(specs)


def dump(specs: tuple[TaskSpec, ...], path: Path) -> None:
    Path(path).write_text(
        "\n".join(spec.to_json() for spec in specs) + "\n", encoding="utf-8"
    )
