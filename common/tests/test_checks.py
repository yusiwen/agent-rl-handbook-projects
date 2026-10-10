"""The five checks, including the ones that must fail.

Every passing check here has a matching failing case: a check that has only ever seen
clean input has not been tested, it has been hoped at.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from common.checks import (
    check_advantage_spread,
    check_boundaries,
    check_mask,
    check_round_stability,
    check_terms,
    main,
    run_all,
)
from common.schema import read_rollouts

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def rows(name: str) -> list:
    return read_rollouts(str(FIXTURES / name))


@contextlib.contextmanager
def quiet():
    """Silence library functions that print, when their output is not the assertion.

    Without this, a test of the *failing* fixture prints a thirty-line failure into the
    gate's log, which reads exactly like the gate itself failing.
    """
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        yield buffer


class HappyPathTests(unittest.TestCase):
    def test_the_good_fixture_passes_every_check(self) -> None:
        with quiet():  # run_all prints; its output is not the assertion
            results = run_all(rows("round-000.jsonl"))
        self.assertEqual([result.name for result in results],
                         ["mask", "boundaries", "terms", "advantage spread"])
        for result in results:
            self.assertTrue(result.ok, result.render())

    def test_term_inventory_is_reported(self) -> None:
        text = check_terms(rows("round-000.jsonl")).render()
        self.assertIn("judge", text)
        self.assertIn("rule", text)

    def test_spread_is_reported_as_a_share_of_groups(self) -> None:
        self.assertIn("2/2 groups usable (100%)", check_advantage_spread(rows("round-000.jsonl")).render())


class MaskCheckTests(unittest.TestCase):
    def test_a_trained_tool_token_fails_the_mask_check_by_name(self) -> None:
        result = check_mask(rows("bad-mask.jsonl"))
        self.assertFalse(result.ok)
        self.assertEqual(result.name, "mask")
        self.assertEqual(len(result.messages), 30)
        self.assertIn("p1-s0", result.messages[0])
        self.assertIn("'tool'", result.messages[0])

    def test_the_other_checks_still_pass_on_the_bad_mask_fixture(self) -> None:
        # The corruption is confined to the mask, which is what makes it a useful test.
        self.assertTrue(check_boundaries(rows("bad-mask.jsonl")).ok)
        self.assertTrue(check_terms(rows("bad-mask.jsonl")).ok)


class BoundaryCheckTests(unittest.TestCase):
    def test_recorded_turn_count_is_compared_with_the_trace(self) -> None:
        good = rows("round-000.jsonl")
        self.assertTrue(check_boundaries(good).ok)
        good[0].turn_count = 5
        result = check_boundaries(good)
        self.assertFalse(result.ok)
        self.assertIn("p1-s0", result.messages[0])

    def test_a_missing_turn_count_is_a_fault_not_a_zero(self) -> None:
        good = rows("round-000.jsonl")
        good[1].turn_count = None
        self.assertIn("no turn_count recorded", check_boundaries(good).messages[0])


class TermCheckTests(unittest.TestCase):
    def test_a_rollout_without_terms_fails(self) -> None:
        good = rows("round-000.jsonl")
        good[0].reward_terms = {}
        result = check_terms(good)
        self.assertFalse(result.ok)
        self.assertIn("p1-s0", result.messages[0])


class SpreadCheckTests(unittest.TestCase):
    def test_flat_groups_are_named(self) -> None:
        good = rows("round-000.jsonl")
        for rollout in good:
            rollout.reward_terms = {"rule": 1.0}  # every attempt identical
        result = check_advantage_spread(good)
        self.assertFalse(result.ok)
        self.assertIn("0/2 groups usable", result.render())
        self.assertIn("flat groups", result.render())

    def test_the_threshold_can_be_demanded_at_one_hundred_percent(self) -> None:
        self.assertTrue(check_advantage_spread(rows("round-000.jsonl"), min_fraction=1.0).ok)
        self.assertFalse(check_advantage_spread(rows("round-000.jsonl"), min_fraction=1.01).ok)


class RoundStabilityTests(unittest.TestCase):
    def test_the_second_fixture_round_moves_rewards(self) -> None:
        result = check_round_stability(rows("round-000.jsonl"), rows("round-001.jsonl"))
        self.assertTrue(result.ok, result.render())
        self.assertIn("changed reward", result.render())

    def test_identical_rounds_mean_the_feedback_never_arrived(self) -> None:
        result = check_round_stability(rows("round-000.jsonl"), rows("round-000.jsonl"))
        self.assertFalse(result.ok)
        self.assertIn("not reaching the batch", result.render())

    def test_new_sample_ids_are_reported_as_a_comparability_problem(self) -> None:
        previous = rows("round-000.jsonl")
        current = rows("round-001.jsonl")
        current[0].prompt_id = "p3"
        result = check_round_stability(previous, current)
        self.assertFalse(result.ok)
        self.assertIn("missing in the later round", result.render())
        self.assertIn("new in the later round", result.render())


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "common.checks", *args],
            cwd=REPO, capture_output=True, text=True, check=False,
        )

    def test_good_fixture_exits_zero(self) -> None:
        result = self.run_cli("--rollouts", str(FIXTURES / "round-000.jsonl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all checks passed", result.stdout)

    def test_bad_mask_fixture_exits_nonzero_and_names_the_layer(self) -> None:
        result = self.run_cli("--rollouts", str(FIXTURES / "bad-mask.jsonl"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL] mask", result.stdout)
        self.assertIn("failed: mask", result.stdout)

    def test_previous_round_turns_on_the_stability_check(self) -> None:
        result = self.run_cli(
            "--rollouts", str(FIXTURES / "round-001.jsonl"),
            "--previous", str(FIXTURES / "round-000.jsonl"),
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("round stability", result.stdout)

    def test_json_output_parses_and_reports_the_same_verdict(self) -> None:
        result = self.run_cli("--rollouts", str(FIXTURES / "bad-mask.jsonl"), "--json")
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["rollouts"], 6)
        verdicts = {check["name"]: check["ok"] for check in payload["checks"]}
        self.assertFalse(verdicts["mask"])
        self.assertTrue(verdicts["boundaries"])

    def test_main_returns_one_rather_than_exiting_so_it_can_be_imported(self) -> None:
        with quiet():  # main prints its report; the assertion is the return value
            self.assertEqual(main(["--rollouts", str(FIXTURES / "round-000.jsonl")]), 0)
            self.assertEqual(main(["--rollouts", str(FIXTURES / "bad-mask.jsonl")]), 1)


if __name__ == "__main__":
    unittest.main()
