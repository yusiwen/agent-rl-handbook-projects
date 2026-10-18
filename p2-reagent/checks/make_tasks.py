"""Generate the Project 2 task set and write it to ``fixtures/tasks.jsonl``.

    python3 p2-reagent/checks/make_tasks.py           # rewrite the fixture
    python3 p2-reagent/checks/make_tasks.py --check   # fail if the committed file would change

The set is generated rather than typed so that the properties hold by construction - a source that
exists, a target that does not, an answer that can be recomputed - and it is committed rather than
generated at run time so that a change to what the project trains on is a visible diff.

The three task ids from the first version (``files-rename-count``, ``files-delete-count``,
``ledger-net-seven``) are reproduced exactly: the loop's fixtures and its measurements name them,
and a task set that silently renames its own tasks makes an old result unattributable.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent.parent))  # for `common`, which the package imports

from p2_reagent import tasksets  # noqa: E402

WORDS = ("draft", "final", "notes", "budget", "report", "photo", "script", "cache", "index", "plan")
EXTS = (".txt", ".md", ".log", ".bak", ".csv")
LETTERS = ("d", "f", "b", "r", "p", "c", "i", "s")


def _files(rng: random.Random, task_id: str, op: str) -> tasksets.TaskSpec:
    """One files task: a directory of three or four names, one operation, one count."""
    while True:
        stems = rng.sample(WORDS, 3)
        names = [stem + rng.choice(EXTS) for stem in stems]
        names = list(dict.fromkeys(names))
        if len(names) < 3:
            continue
        source = names[0]
        if op == "rename":
            target = "final" + rng.choice(EXTS)
            if target in names:
                continue
            spec = tasksets.TaskSpec(
                task_id=task_id,
                family="files",
                op=op,
                world=tuple(sorted(names)),
                source=source,
                target=target,
                count_mode="prefix",
                count_text=source[0],
                expected="0",
            )
        else:
            spec = tasksets.TaskSpec(
                task_id=task_id,
                family="files",
                op=op,
                world=tuple(sorted(names)),
                source=source,
                count_mode="remaining",
                expected="0",
            )
        # Fill in the answer from the specification itself, then let the consistency check
        # confirm it: an expected value typed by hand is a second source of truth.
        spec = tasksets.TaskSpec.from_dict({**spec.as_dict(), "expected": spec.recomputed_answer()})
        if spec.contradiction() is None and 1 <= int(spec.expected) <= 3:
            return spec


def _ledger(rng: random.Random, task_id: str, length: int) -> tasksets.TaskSpec:
    """One ledger task: an ordered plan of movements, and two plausible wrong moves."""
    while True:
        plan = tuple(rng.choice([3, 4, 5, 6, 7, 8, 9, 11, 12, 14]) * rng.choice([1, -1]) for _ in range(length))
        if any(move == 0 for move in plan):
            continue
        expected = sum(plan)
        if expected == 0 or abs(expected) > 20:
            continue
        pool = [m for m in (2, 3, 5, 7, 9, 11, 12, 14) if m not in {abs(p) for p in plan}]
        distractors = tuple(
            [f"add {rng.choice(pool)}", f"sub {rng.choice(pool)}"]
        )
        spec = tasksets.TaskSpec(
            task_id=task_id,
            family="ledger",
            op="sequence",
            world=(),
            expected=str(expected),
            plan=plan,
            distractors=distractors,
        )
        if spec.contradiction() is None:
            return spec


def build() -> tuple[tasksets.TaskSpec, ...]:
    rng = random.Random(20261018)
    specs: list[tasksets.TaskSpec] = [
        tasksets.TaskSpec(
            task_id="files-rename-count",
            family="files",
            op="rename",
            world=("draft.txt", "keep.md", "tmp.log"),
            source="draft.txt",
            target="final.txt",
            count_mode="prefix",
            count_text="f",
            expected="1",
        ),
        tasksets.TaskSpec(
            task_id="files-delete-count",
            family="files",
            op="delete",
            world=("draft.txt", "keep.md", "tmp.log"),
            source="tmp.log",
            count_mode="remaining",
            expected="2",
        ),
        tasksets.TaskSpec(
            task_id="ledger-net-seven",
            family="ledger",
            op="sequence",
            world=(),
            expected="7",
            plan=(12, -5),
            distractors=("add 5", "sub 12"),
        ),
    ]
    for index in range(1, 11):
        specs.append(_files(rng, f"files-r{index:02d}", "rename"))
        specs.append(_files(rng, f"files-d{index:02d}", "delete"))
        specs.append(_ledger(rng, f"ledger-{index:02d}", 2))
        specs.append(_ledger(rng, f"ledger-3s{index:02d}", 3))
    return tuple(specs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="generate the Project 2 task set")
    parser.add_argument("--check", action="store_true", help="fail instead of writing")
    parser.add_argument("--out", default=str(HERE.parent / "fixtures" / "tasks.jsonl"))
    args = parser.parse_args(argv)

    specs = build()
    for spec in specs:
        problem = spec.contradiction()
        if problem is not None:
            print(f"FAIL {spec.task_id}: {problem}")
            return 2
    target = Path(args.out)
    rendered = "\n".join(spec.to_json() for spec in specs) + "\n"
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    if args.check:
        if current != rendered:
            print(f"FAIL {target} is not what the generator produces; run without --check")
            return 2
        print(f"ok {len(specs)} tasks, file matches the generator")
        return 0
    target.write_text(rendered, encoding="utf-8")
    families = {}
    for spec in specs:
        families[spec.family] = families.get(spec.family, 0) + 1
    print(f"wrote {len(specs)} tasks to {target}: {families}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
