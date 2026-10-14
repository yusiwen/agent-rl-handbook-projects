"""Tests for the two Chapter 7 tools: the health report and the contamination check.

    python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests

Both tools are driven as functions, offline, over temporary directories:

* the health report reads a *synthetic* run (a header plus four rollouts), so the numbers it
  prints are known by construction - the point is that the report derives them from the rows
  rather than from anything it was told;
* the contamination check is handed two small JSONL files, so a leak can be planted and seen.

Nothing here touches the real `runs/` tree or `datasets/payloads/`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECKS = REPO / "p1-verifiers" / "checks"
for path in (str(REPO), str(REPO / "p1-verifiers" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.schema import Rollout, Segment  # noqa: E402


def load(name: str):
    """Load a script from checks/ by path: the directory is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = load("check_task_health")
contamination = load("check_contamination")

LONG_ITEM = (
    "Janet's ducks lay sixteen eggs per day. She eats three for breakfast every morning and "
    "bakes muffins with four, then sells the rest at the market for two dollars each."
)

FIXTURE_ROWS = [
    {"id": "t01", "verifier": "numeric", "prompt": "What is the sum of the integers from 1 to 30? End with the number alone on the last line.", "answer": "465"},
    {"id": "t02", "verifier": "numeric", "prompt": "A tap fills a tank in 6 hours and a drain empties it in 9 hours. How long together? End with the number alone on the last line.", "answer": "18"},
    {"id": "t03", "verifier": "logic_puzzle", "prompt": "Three friends each own a differently coloured house: red, green and blue. Ana does not own the red house and Bo does not own the green one. Who owns the red house? End with the name alone on the last line.", "answer": "Cy"},
    {"id": "t04", "verifier": "unit_test", "prompt": "Write a function that returns the n-th Fibonacci number. Put the code in main.py.", "answer": "tests"},
]


def rollout(prompt_id: str, index: int, passes: int, total: int, with_verdict: bool = True) -> Rollout:
    """One attempt; `passes` of `total` attempts score, so the histogram is known up front."""
    scored = index < passes
    return Rollout(
        episode_id=f"{prompt_id}-{index}",
        prompt_id=prompt_id,
        sample_index=index,
        group_id=prompt_id,
        segments=[
            Segment(role="user", text="question", tokens=10),
            Segment(role="model", text="answer", tokens=20),
        ],
        reward_terms={"rule": 1.0 if scored else 0.0},
        mask=[0] * 10 + [1] * 20,
        turn_count=1,
        extras={"passed": scored} if with_verdict else {},
    )


class RunFixture(unittest.TestCase):
    """A synthetic run: t01 never scores, t02 half, t03 always, t04 never attempted."""

    with_verdict = True

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(
            "\n".join(json.dumps(row) for row in FIXTURE_ROWS) + "\n", encoding="utf-8"
        )
        self.run_dir = self.root / "runs" / "synthetic-run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "header.json").write_text(
            json.dumps(
                {
                    "run_id": "synthetic-run",
                    "project": "p1-verifiers",
                    "env_version": "test/v1",
                    "config_hash": "sha256:test",
                    "code_sha": "git:test",
                    "dataset": "tasks.jsonl",
                    "dataset_digest": "sha256:test",
                    "notes": "test model at http://127.0.0.1:8000, temperature=0.7, samples=4",
                }
            ),
            encoding="utf-8",
        )
        rows = (
            [rollout("t01", index, 0, 4, self.with_verdict) for index in range(4)]
            + [rollout("t02", index, 2, 4, self.with_verdict) for index in range(4)]
            + [rollout("t03", index, 4, 4, self.with_verdict) for index in range(4)]
        )
        (self.run_dir / "round-000.jsonl").write_text(
            "".join(row.to_json() + "\n" for row in rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def report(self, *extra: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = health.main(["--tasks", str(self.tasks), "--run", str(self.run_dir), *extra])
        return code, buffer.getvalue()


class HealthReportTests(RunFixture):
    def test_the_histogram_and_band_come_from_the_rows(self) -> None:
        code, text = self.report()
        self.assertEqual(code, 0)
        self.assertIn("0/4", text)
        self.assertIn("<- always wrong: no gradient", text)
        self.assertIn("<- always right: no gradient", text)
        self.assertIn("1 of 3 item(s) inside the band (33%)", text)
        self.assertIn("groups a GRPO update could use: 1 of 3", text)

    def test_the_family_breakdown_counts_the_fixture_and_names_what_never_ran(self) -> None:
        _, text = self.report()
        self.assertIn("arithmetic word problems", text)
        self.assertIn("logic puzzles", text)
        self.assertIn("code (unit tests)", text)
        self.assertIn("(no attempts in this run)", text)

    def test_the_configuration_travels_with_the_report(self) -> None:
        _, text = self.report()
        self.assertIn("git:test", text)
        self.assertIn("sha256:test", text)
        self.assertIn("temperature=0.7", text)
        self.assertIn("scored by: the verifier per-attempt verdict", text)

    def test_the_sample_count_is_stated_beside_the_band(self) -> None:
        """The edge of the band moves with the sample count, so the report has to say it."""
        _, text = self.report()
        self.assertIn("reached at 4 sample(s) means [1, 2, 3] of 4 solved", text)
        self.assertIn("a different sample count moves these edges", text)

    def test_the_contamination_row_is_open_without_an_evaluation_set(self) -> None:
        _, text = self.report()
        self.assertIn("not run (no --eval given)", text)
        self.assertIn("this row stays open", text)

    def test_the_verdict_row_is_left_for_the_reader(self) -> None:
        _, text = self.report()
        self.assertIn("write this before the training run, not after", text)
        self.assertIn("_" * 70, text)

    def test_a_band_share_below_the_threshold_fails_the_run(self) -> None:
        code, text = self.report("--min-band-share", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] band share 33%", text)

    def test_a_band_share_above_the_threshold_passes(self) -> None:
        code, _ = self.report("--min-band-share", "0.2")
        self.assertEqual(code, 0)

    def test_two_rounds_are_reported_separately(self) -> None:
        (self.run_dir / "round-001.jsonl").write_text(
            "".join(row.to_json() + "\n" for row in [rollout("t02", i, 1, 4) for i in range(4)]),
            encoding="utf-8",
        )
        _, text = self.report()
        self.assertIn("describing round 1", text)
        self.assertIn("round 0: 1 of 3 groups usable", text)
        self.assertIn("round 1: 1 of 1 groups usable", text)
        self.assertIn("compare group counts across rounds", text)

    def test_the_json_payload_matches_the_printed_report(self) -> None:
        target = self.root / "health.json"
        code, _ = self.report("--json", str(target))
        self.assertEqual(code, 0)
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["run_id"], "synthetic-run")
        self.assertEqual(payload[0]["inside_band"], 1)
        self.assertEqual(payload[0]["histogram"], {"0": 1, "1": 0, "2": 1, "3": 0, "4": 1})
        self.assertEqual(payload[0]["contamination"], {"ran": False})

    def test_a_missing_run_directory_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            health.main(["--tasks", str(self.tasks), "--run", str(self.root / "nope")])
        self.assertIn("no run directory", str(caught.exception))


class HealthReportWithoutVerdictsTests(RunFixture):
    with_verdict = False

    def test_the_reward_is_used_and_the_report_says_so(self) -> None:
        code, text = self.report()
        self.assertEqual(code, 0)
        self.assertIn("reward >= 1.0 (no verdict field recorded)", text)
        self.assertIn("1 of 3 item(s) inside the band (33%)", text)


class CrossRunTests(unittest.TestCase):
    """Two temperatures are required by the pass criterion, so the join is computed, not implied."""

    def payload(self, run_id: str, solved: dict[str, int], samples: int = 8) -> dict:
        return {"run_id": run_id, "solved": solved, "samples_per_item": samples}

    def test_an_item_in_band_in_one_run_only_is_reported_as_moved(self) -> None:
        line = health.cross_run_line(
            [
                self.payload("a", {"t01": 4, "t02": 8, "t03": 0}),
                self.payload("b", {"t01": 8, "t02": 8, "t03": 0}),
            ]
        )
        self.assertIn("inside the band in every run: 0 of 3", line)
        self.assertIn("inside the band in at least one run: 1 of 3", line)
        self.assertIn("always solved in every run: 1", line)
        self.assertIn("always failed in every run: 1", line)
        self.assertIn("the band moved with the sampler for 1 item(s): t01", line)

    def test_an_item_stable_in_both_runs_is_not_called_moved(self) -> None:
        line = health.cross_run_line(
            [self.payload("a", {"t01": 4}), self.payload("b", {"t01": 5})]
        )
        self.assertIn("inside the band in every run: 1 of 1", line)
        self.assertIn("the band moved with the sampler for 0 item(s)", line)

    def test_the_join_is_over_items_present_in_every_run(self) -> None:
        line = health.cross_run_line(
            [self.payload("a", {"t01": 4, "t02": 4}), self.payload("b", {"t01": 4})]
        )
        self.assertIn("1 item(s) present in all 2 run(s)", line)

    def test_the_report_prints_the_join_when_two_runs_are_given(self) -> None:
        buffer = io.StringIO()
        fixture = RunFixture()
        fixture.setUp()
        try:
            (fixture.run_dir / "round-000.jsonl").write_text(
                "".join(row.to_json() + "\n" for row in [rollout("t02", i, 1, 4) for i in range(4)]),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(buffer):
                code = health.main(
                    [
                        "--tasks", str(fixture.tasks),
                        "--run", str(fixture.run_dir),
                        "--run", str(fixture.run_dir),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("cross-run", buffer.getvalue())
        finally:
            fixture.tearDown()


class ContaminationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(
            json.dumps({"id": "t01", "verifier": "numeric", "prompt": LONG_ITEM, "answer": "1"}) + "\n",
            encoding="utf-8",
        )
        self.eval = self.root / "eval.jsonl"
        self.eval.write_text(
            json.dumps({"question": "A completely different problem about a train and a tunnel."}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *extra: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = contamination.main(["--tasks", str(self.tasks), "--eval", str(self.eval), *extra])
        return code, buffer.getvalue()

    def test_a_clean_pair_exits_zero(self) -> None:
        code, text = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("verdict: nothing overlaps", text)

    def test_a_leak_exits_one_and_is_named(self) -> None:
        self.eval.write_text(json.dumps({"question": LONG_ITEM}) + "\n", encoding="utf-8")
        code, text = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("contaminated: 1", text)
        self.assertIn("t01", text)

    def test_the_digest_of_the_evaluation_file_is_printed(self) -> None:
        _, text = self.run_cli()
        self.assertRegex(text, r"sha256:[0-9a-f]{16}")

    def test_a_custom_field_is_honoured(self) -> None:
        self.eval.write_text(json.dumps({"problem": LONG_ITEM}) + "\n", encoding="utf-8")
        code, _ = self.run_cli("--eval-field", "problem")
        self.assertEqual(code, 1)

    def test_rows_without_the_field_are_counted_not_ignored_silently(self) -> None:
        self.eval.write_text(
            json.dumps({"question": "another unrelated problem entirely"}) + "\n" + json.dumps({"nope": 1}) + "\n",
            encoding="utf-8",
        )
        _, text = self.run_cli()
        self.assertIn("1 evaluation row(s) had no usable 'question' text", text)

    def test_a_missing_evaluation_set_is_refused_with_the_command_that_makes_one(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            contamination.main(
                ["--tasks", str(self.tasks), "--eval", str(self.root / "absent.jsonl")]
            )
        self.assertIn("datasets/rows_jsonl.py", str(caught.exception))

    def test_the_json_report_repeats_the_verdict_and_the_digest(self) -> None:
        target = self.root / "contamination.json"
        self.run_cli("--json", str(target))
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["contaminated"], [])
        self.assertEqual(payload["eval_size"], 1)
        self.assertRegex(payload["eval_digest"], r"^sha256:")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
