"""Chapter 7's fourth row: does this task set overlap its evaluation set?

    python3 p1-verifiers/checks/check_contamination.py \
        --tasks p1-verifiers/fixtures/tasks.jsonl \
        --eval datasets/payloads/gsm8k/rows/test.jsonl

Exit 0 when no task shares a 13-word run with the evaluation set, 1 when one does.

The evaluation file is a JSONL view of a held-out split, written by
`datasets/rows_jsonl.py` (parquet cannot be read by the stdlib-only contract layer, and this
check is not allowed to invent a dependency to work around that).  The digest of that file is
printed with the verdict, because "we checked against the evaluation set" only means something
once the bytes are named.

The field to compare on differs by dataset: GSM8K keeps its problems in ``question``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:  # allow `python3 p1-verifiers/checks/check_contamination.py`
    sys.path.insert(0, str(REPO))

from common.contamination import DEFAULT_NGRAM, compare  # noqa: E402
from common.runs import digest_of  # noqa: E402

DEFAULT_TASKS = HERE.parent / "fixtures" / "tasks.jsonl"


def read_tasks(path: Path) -> list[tuple[str, str]]:
    """``(id, prompt)`` for every task in the fixture."""
    tasks: list[tuple[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        tasks.append((str(row.get("id", lineno)), str(row.get("prompt", ""))))
    return tasks


def read_evaluations(path: Path, field: str, split_label: str) -> tuple[list[tuple[str, str]], int]:
    """``(id, text)`` for every evaluation row that carries the field, plus the skipped count."""
    evaluations: list[tuple[str, str]] = []
    skipped = 0
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get(field)
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        evaluations.append((f"{split_label}[{index}]", text))
    return evaluations, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 p1-verifiers/checks/check_contamination.py",
        description="Chapter 7's contamination check: task set against a held-out evaluation set.",
    )
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="task fixture (JSONL with id/prompt)")
    parser.add_argument("--eval", required=True, help="evaluation set as JSONL (see datasets/rows_jsonl.py)")
    parser.add_argument("--eval-field", default="question", help="field holding the problem text (default: question)")
    parser.add_argument("--split-label", default=None, help="how to name evaluation rows (default: the file's stem)")
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help=f"words per run (default {DEFAULT_NGRAM})")
    parser.add_argument("--json", help="also write the report as JSON here")
    args = parser.parse_args(argv)

    task_path = Path(args.tasks)
    eval_path = Path(args.eval)
    if not task_path.exists():
        raise SystemExit(f"no task fixture at {task_path}")
    if not eval_path.exists():
        raise SystemExit(
            f"no evaluation set at {eval_path}\n"
            "Materialise one with:\n"
            "    python3 datasets/rows_jsonl.py openai/gsm8k --config main --split test"
        )

    tasks = read_tasks(task_path)
    label = args.split_label or eval_path.parent.parent.name or eval_path.stem
    evaluations, skipped = read_evaluations(eval_path, args.eval_field, label)

    try:
        report = compare(
            tasks,
            evaluations,
            ngram=args.ngram,
            eval_source=str(eval_path.relative_to(REPO)) if eval_path.is_relative_to(REPO) else str(eval_path),
            eval_digest=digest_of(eval_path),
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(report.render())
    if skipped:
        print(
            f"\n{skipped} evaluation row(s) had no usable '{args.eval_field}' text and were not compared "
            "- a skipped row cannot contaminate anything, but it also cannot be checked."
        )
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
