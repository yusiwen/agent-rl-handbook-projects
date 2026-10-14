"""Chapter 7's deliverable: the health report for a task set, as a script.

Chapter 7 tells the reader the report is a script and not a document, and that it should be
re-run after every change to the set.  This is that script.  It reads one or more live runs
and prints the five rows of the deliverable table:

1. items, by task family
2. the pass-rate histogram, with the sample count that produced it
3. the share inside the trainable band, and the group count a GRPO update could use
4. the contamination check, when an evaluation set is given
5. a blank verdict line, to be filled in *before* the training run

    python3 p1-verifiers/checks/check_task_health.py --run runs/<id> --eval datasets/payloads/gsm8k/rows/test.jsonl

Two things it refuses to do.  It does not invent samples: if a run holds four attempts per
item, the band's edges are 1/4 and 3/4 and the report says so, because the edge moves with
the sample count (Chapter 7, Lesson 4).  And it does not write the verdict, because a verdict
written after the training run is a rationalisation.

Exit status is 0 unless ``--min-band-share`` is given and the set falls short of it.  A
report is not a gate: the reader decides what share is good enough.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.contamination import DEFAULT_NGRAM, compare  # noqa: E402
from common.runs import SPREAD_EPSILON, digest_of, spread_per_group  # noqa: E402
from common.schema import Rollout, read_rollouts  # noqa: E402

DEFAULT_TASKS = HERE.parent / "fixtures" / "tasks.jsonl"

#: The verifier names the fixture uses, and the task family each one stands for.  The family
#: is what the reader counts; the verifier is the thing that scores it.
FAMILIES = {
    "exact_match": "exact answer",
    "numeric": "arithmetic word problems",
    "logic_puzzle": "logic puzzles",
    "unit_test": "code (unit tests)",
}
UNKNOWN_FAMILY = "unmapped verifier"

#: The band Chapter 7 uses: a pass rate strictly inside these two edges.  Both are excluded
#: because a set is either always solved or never solved at those points.
BAND_LOW = 0.1
BAND_HIGH = 0.9


def read_fixture(path: Path) -> dict[str, dict[str, str]]:
    """``task id -> {family, prompt}``, read from the fixture the run was made from.

    The prompt is taken from the fixture rather than rebuilt from a trace, because the
    contamination row is a statement about the *task set*, and the fixture is where the task
    set lives.  Rebuilding text out of segments would compare whatever the harness happened to
    record, and would silently compare nothing if the harness changed.
    """
    fixture: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fixture[str(row.get("id"))] = {
            "family": FAMILIES.get(str(row.get("verifier")), UNKNOWN_FAMILY),
            "prompt": str(row.get("prompt", "")),
        }
    return fixture


def passed(rollout: Rollout) -> bool:
    """Whether one attempt scored, preferring the verifier's own per-attempt verdict."""
    if "passed" in rollout.extras:
        return bool(rollout.extras["passed"])
    return rollout.reward() >= 1.0 - SPREAD_EPSILON


def rounds_of(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("round-*.jsonl"))


def load_runs(run_dir: Path) -> tuple[dict, list[Rollout], list[list[Rollout]]]:
    header = json.loads((run_dir / "header.json").read_text(encoding="utf-8"))
    per_round = [read_rollouts(str(path)) for path in rounds_of(run_dir)]
    if not per_round:
        raise SystemExit(f"{run_dir} holds no round-*.jsonl files")
    return header, [row for round_rows in per_round for row in round_rows], per_round


def bar(count: int, scale: int = 1) -> str:
    return "#" * max(0, count // max(scale, 1))


def render_run(
    run_dir: Path,
    fixture: dict[str, dict[str, str]],
    eval_path: Path | None,
    eval_field: str,
    ngram: int,
) -> tuple[list[str], dict]:
    header, _all_rows, per_round = load_runs(run_dir)
    # The report describes one batch, and a batch is one round.  Pooling rounds would double the
    # sample count behind every pass rate and move the edge of the band, which is the one number
    # Chapter 7 insists must travel with its sample count.  Earlier rounds are compared below, by
    # group counts.
    rollouts: list[Rollout] = per_round[-1]
    attempts: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        attempts[rollout.prompt_id].append(rollout)

    sample_counts = {len(rows) for rows in attempts.values()}
    samples = f"{min(sample_counts)}-{max(sample_counts)}" if len(sample_counts) > 1 else str(min(sample_counts))
    verdict_field = all("passed" in row.extras for row in rollouts)

    lines = [
        f"task health: {header['run_id']}",
        f"  project {header['project']}  code {header['code_sha']}  config {header['config_hash']}",
        f"  configuration: {header.get('notes', '') or '(no notes recorded)'}",
        f"  {len(rollouts)} attempt(s) over {len(attempts)} item(s), {samples} sample(s) each, "
        + (f"{len(per_round)} round(s), describing round {len(per_round) - 1}" if len(per_round) > 1 else "1 round"),
        f"  scored by: {'the verifier per-attempt verdict' if verdict_field else 'reward >= 1.0 (no verdict field recorded)'}",
        "",
    ]

    # ------------------------------------------------------------------ row 1
    family_counts: Counter[str] = Counter(
        entry["family"] for entry in fixture.values()
    )
    present = Counter(fixture.get(prompt_id, {}).get("family", UNKNOWN_FAMILY) for prompt_id in attempts)
    lines.append("row 1  items by task family")
    for family in sorted(family_counts):
        counted = present.get(family, 0)
        note = "" if counted else "   (no attempts in this run)"
        lines.append(f"  {family:<28} {family_counts[family]:>3}{note}")
    lines.append(f"  {'total':<28} {sum(family_counts.values()):>3}")
    missing = sorted(set(attempts) - set(fixture))
    if missing:
        lines.append(f"  attempted but not in the fixture: {', '.join(missing)}")
    lines.append("")

    # ------------------------------------------------------------------ rows 2 and 3
    solved = {
        prompt_id: sum(1 for row in rows if passed(row))
        for prompt_id, rows in sorted(attempts.items())
    }
    counts = Counter(solved.values())
    top = max(counts.values()) if counts else 1
    scale = max(top // 40, 1)
    per_item = max((len(rows) for rows in attempts.values()), default=0)
    lines.append(f"row 2  pass-rate histogram (attempts solved, out of {per_item})")
    for hits in range(per_item + 1):
        count = counts.get(hits, 0)
        edge = ""
        if hits == 0:
            edge = "  <- always wrong: no gradient"
        elif hits == per_item:
            edge = "  <- always right: no gradient"
        lines.append(f"  {hits:>2}/{per_item}  {bar(count, scale):<40} {count:>3}{edge}")
    lines.append("")

    inside = [
        prompt_id
        for prompt_id, hits in solved.items()
        if BAND_LOW < hits / per_item < BAND_HIGH
    ] if per_item else []
    shares = [hits / per_item for hits in solved.values()] if per_item else []
    spreads = spread_per_group(rollouts)
    usable = [group for group, value in spreads.items() if value > SPREAD_EPSILON]
    lines.append("row 3  inside the trainable band")
    lines.append(
        f"  band {BAND_LOW}-{BAND_HIGH} reached at {per_item} sample(s) means "
        f"{[hits for hits in range(per_item + 1) if BAND_LOW < hits / per_item < BAND_HIGH]} "
        f"of {per_item} solved"
    )
    if solved:
        lines.append(
            f"  {len(inside)} of {len(solved)} item(s) inside the band ({len(inside) / len(solved):.0%})"
            + (f"; mean pass rate {sum(shares) / len(shares):.0%}" if shares else "")
        )
    lines.append(
        f"  groups a GRPO update could use: {len(usable)} of {len(spreads)} "
        f"({len(usable) / len(spreads):.0%})" if spreads else "  no groups"
    )
    if len(per_round) > 1:
        for index, round_rows in enumerate(per_round):
            round_spreads = spread_per_group(round_rows)
            round_usable = sum(1 for value in round_spreads.values() if value > SPREAD_EPSILON)
            lines.append(f"  round {index}: {round_usable} of {len(round_spreads)} groups usable")
        lines.append("  (compare group counts across rounds, not rates: Chapter 7, Lesson 4)")
    lines.append(
        "  a different sample count moves these edges: the band is a property of the policy, "
        "the sampler and the sample count"
    )
    lines.append("")

    # ------------------------------------------------------------------ row 4
    lines.append("row 4  contamination against the evaluation set")
    contamination: dict = {"ran": False}
    if eval_path is None:
        lines.append("  not run (no --eval given): this row stays open, and the set is not in RL until it closes")
    elif not eval_path.exists():
        lines.append(f"  not run: no evaluation set at {eval_path}")
        lines.append("  materialise one with: python3 datasets/rows_jsonl.py openai/gsm8k --config main --split test")
    else:
        evaluations: list[tuple[str, str]] = []
        for index, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            text = json.loads(line).get(eval_field)
            if isinstance(text, str) and text.strip():
                evaluations.append((f"{eval_path.parent.parent.name}[{index}]", text))
        report = compare(
            [(prompt_id, fixture.get(prompt_id, {}).get("prompt", "")) for prompt_id in sorted(attempts)],
            evaluations,
            ngram=ngram,
            eval_source=str(eval_path),
            eval_digest=digest_of(eval_path),
        )
        contamination = report.to_dict()
        lines.append(
            f"  {report.eval_size} item(s) compared ({report.eval_digest}), {ngram}-word runs: "
            f"{len(report.contaminated)} contaminated, {len(report.undecided)} undecided"
        )
        lines.append(f"  {report.verdict_line()}")
    lines.append("")

    # ------------------------------------------------------------------ row 5
    lines.append("row 5  verdict (write this before the training run, not after)")
    lines.append("  " + "_" * 70)

    return lines, {
        "run_id": header["run_id"],
        "code_sha": header["code_sha"],
        "config_hash": header["config_hash"],
        "configuration": header.get("notes", ""),
        "items": len(attempts),
        "samples_per_item": per_item,
        "rounds": len(per_round),
        "families": dict(family_counts),
        "histogram": {str(hits): counts.get(hits, 0) for hits in range(per_item + 1)},
        "inside_band": len(inside),
        "mean_pass_rate": (sum(shares) / len(shares)) if shares else 0.0,
        "usable_groups": len(usable),
        "groups": len(spreads),
        "contamination": contamination,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 p1-verifiers/checks/check_task_health.py",
        description="Chapter 7's deliverable: the health report for a task set, from a live run.",
    )
    parser.add_argument("--run", action="append", required=True, help="a run directory (repeat for several)")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="task fixture, for the family breakdown")
    parser.add_argument("--eval", help="evaluation set as JSONL, for the contamination row")
    parser.add_argument("--eval-field", default="question", help="field holding the problem text")
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help=f"words per run (default {DEFAULT_NGRAM})")
    parser.add_argument(
        "--min-band-share",
        type=float,
        default=0.0,
        help="fail if fewer than this share of items sit in the band (default: report only)",
    )
    parser.add_argument("--json", help="also write the report as JSON here")
    args = parser.parse_args(argv)

    fixture = read_fixture(Path(args.tasks))
    eval_path = Path(args.eval) if args.eval else None

    payloads = []
    failing = False
    for index, run in enumerate(args.run):
        run_dir = Path(run)
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory at {run_dir}")
        lines, payload = render_run(run_dir, fixture, eval_path, args.eval_field, args.ngram)
        if index:
            print()
        print("\n".join(lines))
        payloads.append(payload)
        items = payload["items"] or 1
        if payload["inside_band"] / items < args.min_band_share:
            failing = True
            print(
                f"\n[FAIL] band share {payload['inside_band'] / items:.0%} is below "
                f"--min-band-share {args.min_band_share:.0%}"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(payloads, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if failing else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
