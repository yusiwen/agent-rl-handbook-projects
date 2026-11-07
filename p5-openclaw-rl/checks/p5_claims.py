"""The numbers Chapter 10's Project 5 lessons quote, recomputed from the run records.

The records are committed (`fixtures/runs/`): three loop summaries with their per-sample and
per-round rows, and three greedy evaluations.  Everything here is arithmetic over those files, so a
number in the prose that the records do not support fails the gate rather than surviving a proofread.

    python3 p5-openclaw-rl/checks/p5_claims.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "fixtures" / "runs"
sys.path.insert(0, str(HERE.parent / "src"))

from p5_openclaw import credit  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SummaryTests(unittest.TestCase):
    def test_every_summary_agrees_with_its_round_log(self):
        for run in ("sync", "async1w", "async3w"):
            with self.subTest(run=run):
                summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
                updates = rows(RUNS / run / "updates.jsonl")
                recomputed = credit.summarise(updates, seconds=summary["seconds"])
                self.assertEqual(recomputed["rounds"], summary["rounds"])
                self.assertEqual(recomputed["samples"], summary["samples"])
                self.assertEqual(recomputed["used"], summary["used"])
                self.assertEqual(recomputed["dropped"], summary["dropped"])
                self.assertEqual(recomputed["max_staleness"], summary["max_staleness"])

    def test_samples_per_minute_is_a_division(self):
        for run in ("sync", "async1w", "async3w"):
            with self.subTest(run=run):
                summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
                expected = summary["samples"] / (summary["seconds"] / 60)
                # the summary divides unrounded seconds; the log keeps one decimal
                self.assertAlmostEqual(summary["samples_per_minute"], expected, delta=0.2)

    def test_the_sample_log_matches_the_round_log(self):
        for run in ("sync", "async1w", "async3w"):
            with self.subTest(run=run):
                summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
                samples = rows(RUNS / run / "samples.jsonl")
                updates = rows(RUNS / run / "updates.jsonl")
                self.assertEqual(len(samples), summary["samples"])
                self.assertEqual(
                    sum(1 for row in samples if row["solved"]),
                    sum(row["solved_in_batch"] for row in updates),
                )


class FindingTests(unittest.TestCase):
    """The two claims the lessons make about what this world can and cannot teach."""

    def test_the_two_failing_tasks_fail_deterministically(self):
        samples = rows(RUNS / "sync" / "samples.jsonl")
        by_task: dict[str, list[dict]] = {}
        for row in samples:
            by_task.setdefault(row["task_id"], []).append(row)
        self.assertEqual(len(by_task), 9)
        unsolved = {task: sum(1 for row in group if row["solved"]) for task, group in by_task.items()}
        self.assertEqual(sorted(task for task, solved in unsolved.items() if solved == 0),
                         ["total-revenue", "units-gap"])
        # every solved task was solved every time: no group had spread, which is why `used` is 0
        self.assertTrue(all(solved in (0, len(by_task[task])) for task, solved in unsolved.items()))
        self.assertEqual(len({row["task_id"] for row in samples if row["solved"]}), 7)

    def test_the_turn_budget_was_never_the_constraint(self):
        for name in ("eval-baseline", "eval-sync", "eval-async3w"):
            evaluation = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation["summary"]["turn_budget_exhausted"], 0)

    def test_the_evaluations_agree_across_the_three_policies(self):
        baseline = json.loads((RUNS / "eval-baseline.json").read_text(encoding="utf-8"))["summary"]
        sync_ = json.loads((RUNS / "eval-sync.json").read_text(encoding="utf-8"))["summary"]
        async_ = json.loads((RUNS / "eval-async3w.json").read_text(encoding="utf-8"))["summary"]
        self.assertEqual({baseline["solved"], sync_["solved"], async_["solved"]}, {7})
        self.assertEqual({baseline["tasks"], sync_["tasks"], async_["tasks"]}, {9})

    def test_throughput_at_one_worker_is_the_same_in_both_modes(self):
        sync_ = json.loads((RUNS / "sync" / "summary.json").read_text(encoding="utf-8"))
        async_ = json.loads((RUNS / "async1w" / "summary.json").read_text(encoding="utf-8"))
        # async's 3x figure comes from three workers, not from overlapping training with sampling
        self.assertLess(abs(sync_["samples_per_minute"] - async_["samples_per_minute"]), 3.0)
        three = json.loads((RUNS / "async3w" / "summary.json").read_text(encoding="utf-8"))
        self.assertGreater(three["samples_per_minute"], 2.5 * sync_["samples_per_minute"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
