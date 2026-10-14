"""Tests for P1: extraction, verification, task loading, and the loop.

    python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests

The directory is not a valid Python package name (it has a hyphen), so its own path is
the discovery root and this file puts the repository and ``p1-verifiers/src`` on
``sys.path`` itself.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "p1-verifiers" / "src"
for path in (str(REPO), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.verifier import MissingVerifier  # noqa: E402

from p1_verifiers import (  # noqa: E402
    DEFAULT_ATTEMPT_FILE,
    DEFAULT_TEST_TIMEOUT_S,
    UNIMPLEMENTED,
    DockerSandbox,
    ExactMatchVerifier,
    FixtureResponder,
    LocalSandbox,
    LogicPuzzleVerifier,
    NumericVerifier,
    Provenance,
    SandboxResult,
    SandboxUnavailable,
    Task,
    TaskSetError,
    UnitTestVerifier,
    attempt_source,
    default_registry,
    load_tasks,
    normalise,
    partition_by_coverage,
    pass_rate,
    report,
    run_episode,
    run_taskset,
)
from p1_verifiers.verifiers import (  # noqa: E402
    FORM_PREFIX,
    constraint_failure,
    declared_number,
    final_answer,
    numbers_in,
    parse_assignment,
)

FIXTURES = REPO / "p1-verifiers" / "fixtures"


def fixture_rows():
    """The fixture run in one place: two attempts for every task the suite can score.

    The sandbox is local and says why: the fixture's attempts are written in this repository,
    so the offline gate must not need a container runtime to score them.
    """
    tasks = load_tasks(FIXTURES / "tasks.jsonl")
    covered, _ = partition_by_coverage(tasks, default_registry())
    return run_taskset(
        covered,
        FixtureResponder.from_jsonl(FIXTURES / "responses.jsonl"),
        default_registry(sandbox=LocalSandbox(why="the fixture's attempts are in this repository")),
        samples_per_task=2,
    )


class ExtractionTests(unittest.TestCase):
    def test_last_non_empty_line_wins(self) -> None:
        self.assertEqual(final_answer("reasoning here\n\n391"), "391")

    def test_a_single_line_is_its_own_answer(self) -> None:
        self.assertEqual(final_answer("42"), "42")

    def test_a_trailing_full_stop_is_dropped(self) -> None:
        self.assertEqual(final_answer("The answer is 12.\n\n12."), "12")

    def test_surrounding_whitespace_is_dropped(self) -> None:
        self.assertEqual(final_answer("  12  \n"), "12")

    def test_an_empty_response_extracts_nothing_rather_than_raising(self) -> None:
        self.assertEqual(final_answer(""), "")
        self.assertEqual(final_answer("\n\n   \n"), "")

    def test_it_does_not_understand_equations(self) -> None:
        # Documented behaviour, not an accident: 'x = 12' is not the answer '12'.
        self.assertEqual(final_answer("x + 7 = 19\n\nx = 12"), "x = 12")


class NormaliseTests(unittest.TestCase):
    def test_case_and_spacing_fold(self) -> None:
        self.assertEqual(normalise("  Answer   Is   12 "), "answer is 12")

    def test_trailing_punctuation_folds(self) -> None:
        self.assertEqual(normalise("12."), "12")

    def test_nothing_else_folds(self) -> None:
        self.assertNotEqual(normalise("1024.0"), normalise("1024"))
        self.assertNotEqual(normalise("1,000"), normalise("1000"))


class NumbersInTests(unittest.TestCase):
    def test_every_number_in_the_order_it_appears(self) -> None:
        self.assertEqual(numbers_in("3 litres a minute for 12 minutes is 36"), [3.0, 12.0, 36.0])

    def test_thousands_separators_exponents_and_signs_parse(self) -> None:
        self.assertEqual(numbers_in("a population of 1,000,000"), [1000000.0])
        self.assertEqual(numbers_in("about 1e3 tokens"), [1000.0])
        self.assertEqual(numbers_in("x = -20"), [-20.0])

    def test_no_number_is_an_empty_list_not_an_error(self) -> None:
        self.assertEqual(numbers_in("I do not know."), [])
        self.assertEqual(numbers_in(""), [])


class DeclaredNumberTests(unittest.TestCase):
    """The answer is *declared*: the last line, carrying exactly one number.

    This rule replaced "the last number anywhere in the response" after a live run showed both
    of that rule's failure modes on real output (see ``NumericVerifierTests`` below).
    """

    def test_a_bare_number_on_the_last_line_declares_it(self) -> None:
        self.assertEqual(declared_number("reasoning here\n\n391"), (391.0, ""))

    def test_a_single_line_response_declares_its_one_number(self) -> None:
        self.assertEqual(declared_number("42")[0], 42.0)

    def test_a_number_with_its_unit_still_declares_it(self) -> None:
        # Units stay ignored, which is the documented hedge: '150 minutes' is the answer 150.
        self.assertEqual(declared_number("150 minutes")[0], 150.0)

    def test_a_prose_line_asserting_two_quantities_declares_nothing(self) -> None:
        # The measured false negative: the right answer, in one line, read as no answer at all.
        value, reason = declared_number("There are 150 minutes in 2.5 hours.")
        self.assertIsNone(value)
        self.assertIn("states 2 numbers", reason)

    def test_the_last_line_is_the_declaration_even_when_earlier_lines_have_numbers(self) -> None:
        self.assertEqual(declared_number("5 - 2 = 3 litres a minute\n\n36")[0], 36.0)

    def test_a_final_line_with_no_number_says_so(self) -> None:
        value, reason = declared_number("The tank holds 36 litres.\n\nThat is the result.")
        self.assertIsNone(value)
        self.assertIn("final line states no number", reason)

    def test_a_response_with_no_number_at_all_says_so(self) -> None:
        for response in ("I do not know.", ""):
            with self.subTest(response=response):
                value, reason = declared_number(response)
                self.assertIsNone(value)
                self.assertTrue(reason)


class ExactMatchTests(unittest.TestCase):
    verifier = ExactMatchVerifier()

    def test_a_correct_attempt_scores_one(self) -> None:
        outcome = self.verifier.verify({"answer": "391"}, "17 * 23 = 391\n\n391")
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)

    def test_a_wrong_attempt_scores_zero_and_says_what_was_extracted(self) -> None:
        outcome = self.verifier.verify({"answer": "150"}, "2.5 hours is 150 minutes.\n\n150 minutes")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("150 minutes", outcome.detail)

    def test_it_refuses_a_different_surface_form_for_the_same_number(self) -> None:
        # The case that motivates NumericVerifier: same value, different text.
        outcome = self.verifier.verify({"answer": "1024"}, "2^10 is 1024.\n\n1024.0")
        self.assertFalse(outcome.passed)

    def test_can_verify_needs_a_string_answer(self) -> None:
        self.assertTrue(self.verifier.can_verify({"answer": "1"}))
        self.assertFalse(self.verifier.can_verify({"answer": 1}))
        self.assertFalse(self.verifier.can_verify({}))

    def test_the_reward_term_is_named(self) -> None:
        outcome = self.verifier.verify({"answer": "1"}, "1")
        self.assertEqual(outcome.as_terms("rule"), {"rule": 1.0})


class NumericVerifierTests(unittest.TestCase):
    verifier = NumericVerifier()

    @staticmethod
    def task(answer: str = "3.14", tolerance: float = 0.01) -> dict[str, object]:
        return {"id": "n", "answer": answer, "tolerance": tolerance}

    def test_the_same_value_scores_one_whatever_its_surface_form(self) -> None:
        for response in ("3.14", "3.140", "the value is 3.14 exactly", "pi, to two places:\n\n3.14"):
            with self.subTest(response=response):
                self.assertTrue(self.verifier.verify(self.task(), response).passed)

    def test_a_value_from_the_other_side_of_the_exact_match_failure_passes(self) -> None:
        # '1024.0' is what ExactMatchVerifier refuses; a tolerance accepts it.
        self.assertTrue(self.verifier.verify(self.task("1024", 0.5), "2^10 is 1024.\n\n1024.0").passed)

    def test_a_value_outside_the_tolerance_fails_and_says_by_how_much(self) -> None:
        outcome = self.verifier.verify(self.task(), "pi rounds to 3.1 when you keep two digits.\n\n3.1")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("off by 0.04", outcome.detail)
        self.assertTrue(outcome.detail.startswith(f"{FORM_PREFIX}ok"))

    def test_the_tolerance_is_a_knife_edge_at_floating_point(self) -> None:
        # Measured, not assumed: abs(3.15 - 3.14) really is 0.009999999999999787, which is
        # *inside* a tolerance of 0.01. Both sides are binary floats, so a value that looks
        # exactly one tolerance away can land either way; a task whose boundary matters needs
        # a tolerance that is not the answer's own last digit.
        self.assertLess(abs(3.15 - 3.14), 0.01)
        self.assertTrue(self.verifier.verify(self.task(), "3.15").passed)
        self.assertFalse(self.verifier.verify(self.task(), "3.16").passed)

    def test_units_are_ignored_in_both_directions(self) -> None:
        # The hedge this buys: '150 minutes' is the answer 150. The hedge's cost is right
        # next to it: '150 kilometres' is also the answer 150. Tasks where the unit carries
        # meaning want exact_match, which compares text.
        self.assertTrue(self.verifier.verify(self.task("150", 0.0), "that is 150 minutes").passed)
        self.assertTrue(self.verifier.verify(self.task("150", 0.0), "that is 150 kilometres").passed)
        self.assertFalse(ExactMatchVerifier().verify({"answer": "150"}, "150 kilometres").passed)

    def test_a_response_with_no_number_fails_without_raising(self) -> None:
        outcome = self.verifier.verify(self.task(), "I would rather not say.")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("contains no number", outcome.detail)
        self.assertTrue(outcome.detail.startswith(f"{FORM_PREFIX}no"))

    # --- the format gate ---------------------------------------------------------------
    # Two live failure modes are pinned here, because both looked like arithmetic from the
    # reward curve and neither was.

    def test_a_correct_answer_in_prose_scores_zero_and_names_the_value_it_saw(self) -> None:
        # Measured live: 'There are 150 minutes in 2.5 hours.' is the right answer, and under
        # 'the last number anywhere' it scored 0.0 on all four samples, because 2.5 came last.
        # It still scores 0.0 - but the reason is stated, and the detail names the value it can
        # see, so a flat group can be attributed instead of guessed at.
        outcome = self.verifier.verify(self.task("150", 0.0), "There are 150 minutes in 2.5 hours.")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("states 2 numbers", outcome.detail)
        self.assertIn("the expected value is also in that line", outcome.detail)

    def test_the_detail_does_not_claim_the_arithmetic_was_right(self) -> None:
        # A format experiment settled this: forced to declare one number, the model was wrong in
        # most of the attempts where the hint fired.  The verifier can see that the expected value
        # is in the line; it cannot see that the response meant it, so it must not say so.
        outcome = self.verifier.verify(self.task("150", 0.0), "There are 150 minutes in 2.5 hours.")
        self.assertNotIn("arithmetic looks right", outcome.detail)
        self.assertNotIn("arithmetic is right", outcome.detail)

    def test_identical_answers_score_the_same_however_the_sentence_ends(self) -> None:
        # The measured false variance: four samples of a tank problem all computed 36, and the
        # two whose last line ended at the minutes scored 0.0 while the two that ended at the
        # litres scored 1.0. Both halves now score the same, for the same stated reason: that
        # line asserts two quantities, so nothing was declared.
        ends_after_the_minutes = "3 litres a minute.\n\nSo, after 12 minutes, the tank contains 36 litres."
        ends_at_the_minutes = "3 litres a minute.\n\nSo, the tank contains 36 litres after 12 minutes."
        first = self.verifier.verify(self.task("36", 0.0), ends_after_the_minutes)
        second = self.verifier.verify(self.task("36", 0.0), ends_at_the_minutes)
        self.assertEqual((first.passed, first.score), (second.passed, second.score))
        self.assertFalse(first.passed)
        self.assertTrue(first.detail.startswith(f"{FORM_PREFIX}no"))

    def test_the_boundary_of_the_rule_is_a_stated_one(self) -> None:
        # Where the rule draws its line, written down rather than discovered: a final line with
        # the value and its unit declares that value - add a second quantity and it declares
        # nothing. That is the price of a rule a reader can check; the alternative is a cleverer
        # extractor, which is a new failure mode nobody can see in a reward curve.
        declared = self.verifier.verify(self.task("36", 0.0), "3 litres a minute.\n\n36 litres")
        undeclared = self.verifier.verify(
            self.task("36", 0.0), "3 litres a minute.\n\n36 litres after 12 minutes"
        )
        self.assertTrue(declared.passed)
        self.assertFalse(undeclared.passed)
        self.assertIn("states 2 numbers", undeclared.detail)

    def test_a_declared_answer_is_scored_whatever_the_prose_before_it_says(self) -> None:
        outcome = self.verifier.verify(self.task("36", 0.0), "3 litres a minute for 12 minutes.\n\n36")
        self.assertTrue(outcome.passed)
        self.assertTrue(outcome.detail.startswith(f"{FORM_PREFIX}ok"))

    def test_can_verify_needs_an_answer_and_a_tolerance(self) -> None:
        self.assertTrue(self.verifier.can_verify(self.task()))
        self.assertFalse(self.verifier.can_verify({"answer": "3.14"}))
        self.assertFalse(self.verifier.can_verify({"tolerance": 0.01}))
        self.assertFalse(self.verifier.can_verify({}))

    def test_a_task_whose_answer_is_not_a_number_is_an_error_not_a_zero(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.verifier.verify(self.task(answer="three point one four"), "3.14")
        self.assertIn("not a number", str(ctx.exception))

    def test_a_negative_tolerance_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.verifier.verify(self.task(tolerance=-0.01), "3.14")
        self.assertIn("negative tolerance", str(ctx.exception))

    def test_the_reward_term_is_named(self) -> None:
        self.assertEqual(self.verifier.verify(self.task(), "3.14").as_terms("rule"), {"rule": 1.0})


class AttemptSourceTests(unittest.TestCase):
    def test_a_plain_response_is_its_own_source(self) -> None:
        self.assertEqual(attempt_source("def add(a, b):\n    return a + b"),
                         "def add(a, b):\n    return a + b")

    def test_a_fenced_response_loses_the_fence(self) -> None:
        # The fence is presentation; writing it into the file would score the wrapper.
        self.assertEqual(attempt_source("```python\ndef add(a, b):\n    return a + b\n```"),
                         "def add(a, b):\n    return a + b")

    def test_prose_around_the_fence_is_left_behind(self) -> None:
        response = "Here it is:\n\n```python\ndef add(a, b):\n    return a + b\n```\nThat adds them."
        self.assertEqual(attempt_source(response), "def add(a, b):\n    return a + b")

    def test_the_last_block_wins(self) -> None:
        response = "```\nfirst = 1\n```\nand then:\n```\nsecond = 2\n```"
        self.assertEqual(attempt_source(response), "second = 2")

    def test_an_empty_response_has_no_source(self) -> None:
        self.assertEqual(attempt_source("   \n\n"), "")


VARIABLES = ["alice", "bob", "carol"]


class ParseAssignmentTests(unittest.TestCase):
    def test_a_json_object_is_read(self) -> None:
        self.assertEqual(parse_assignment('{"alice": "green", "bob": "blue"}', VARIABLES),
                         {"alice": "green", "bob": "blue"})

    def test_the_last_json_object_wins(self) -> None:
        response = '{"alice": "red", "bob": "blue"}\nStill not sure:\n{"alice": "green", "bob": "blue"}'
        self.assertEqual(parse_assignment(response, VARIABLES),
                         {"alice": "green", "bob": "blue"})

    def test_name_equals_value_lines_are_read(self) -> None:
        response = "Reasoning first.\n\nalice = green\nbob = blue"
        self.assertEqual(parse_assignment(response, VARIABLES), {"alice": "green", "bob": "blue"})

    def test_a_colon_line_reads_the_same_way(self) -> None:
        self.assertEqual(parse_assignment("alice: green", VARIABLES), {"alice": "green"})

    def test_a_later_line_overrides_an_earlier_one(self) -> None:
        response = "alice = red\nbob = blue\n\nalice = green"
        self.assertEqual(parse_assignment(response, VARIABLES)["alice"], "green")

    def test_only_declared_variables_survive(self) -> None:
        # Prose and undeclared names are left out: the task says what the answer is about.
        response = "note: this took a while\nanswer = 3\nalice = green"
        self.assertEqual(parse_assignment(response, VARIABLES), {"alice": "green"})

    def test_case_and_spacing_fold_on_both_sides(self) -> None:
        self.assertEqual(parse_assignment('{"Alice": " Green "}', VARIABLES), {"alice": "green"})

    def test_braces_that_are_not_json_are_skipped(self) -> None:
        response = "{not json}\n\nalice = green"
        self.assertEqual(parse_assignment(response, VARIABLES), {"alice": "green"})

    def test_no_assignment_anywhere_gives_an_empty_mapping(self) -> None:
        self.assertEqual(parse_assignment("I would rather not say.", VARIABLES), {})
        self.assertEqual(parse_assignment("", VARIABLES), {})
        self.assertEqual(parse_assignment('{"dave": "green"}', VARIABLES), {})


class LogicPuzzleVerifierTests(unittest.TestCase):
    verifier = LogicPuzzleVerifier()
    SOLUTION = {
        "variables": VARIABLES,
        "constraints": [
            {"id": "c1", "kind": "distinct"},
            {"id": "c2", "kind": "equals", "var": "bob", "value": "blue"},
            {"id": "c3", "kind": "not_equals", "var": "alice", "value": "red"},
        ],
    }

    def verify(self, response: str, solution: dict[str, object] | None = None):
        return self.verifier.verify({"id": "p", "solution": solution or self.SOLUTION}, response)

    def test_a_complete_consistent_assignment_scores_one(self) -> None:
        outcome = self.verify("alice = green\nbob = blue\ncarol = red")
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)
        self.assertIn("all 3 constraints hold", outcome.detail)

    def test_a_near_miss_scores_zero_and_says_which_constraint_failed(self) -> None:
        # Two of three constraints hold, so the report has to read as a near miss: the failed
        # constraint is named, with the reason, even though the score is a hard zero.
        outcome = self.verify('{"alice": "blue", "bob": "blue", "carol": "red"}')
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("satisfied 2 of 3 constraints", outcome.detail)
        self.assertIn("c1", outcome.detail)
        self.assertIn("both 'blue'", outcome.detail)

    def test_a_missing_variable_fails_rather_than_passing_vacuously(self) -> None:
        outcome = self.verify("alice = green\nbob = blue")
        self.assertFalse(outcome.passed)
        self.assertIn("carol missing from the answer", outcome.detail)

    def test_a_wrong_value_is_reported_with_what_was_actually_said(self) -> None:
        outcome = self.verify("alice = green\nbob = green\ncarol = red")
        self.assertIn("bob must be 'blue', got 'green'", outcome.detail)

    def test_a_violated_negative_constraint_says_so(self) -> None:
        outcome = self.verify("alice = red\nbob = green\ncarol = blue")
        self.assertIn("alice must not be 'red', and is", outcome.detail)

    def test_a_response_with_no_assignment_is_reported_as_such(self) -> None:
        outcome = self.verify("I cannot decide between two colourings.")
        self.assertFalse(outcome.passed)
        self.assertIn("contains no assignment", outcome.detail)

    def test_values_are_compared_case_insensitively(self) -> None:
        self.assertTrue(self.verify("Alice = Green\nBob = Blue\nCarol = Red").passed)

    def test_can_verify_needs_a_solution_object(self) -> None:
        self.assertTrue(self.verifier.can_verify({"solution": self.SOLUTION}))
        self.assertTrue(self.verifier.can_verify({"solution": {}}))
        self.assertFalse(self.verifier.can_verify({}))
        self.assertFalse(self.verifier.can_verify({"solution": "distinct"}))

    def test_a_solution_without_variables_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.verify("alice = green", solution={"constraints": [{"kind": "distinct"}]})
        self.assertIn("without 'variables'", str(ctx.exception))

    def test_a_solution_without_constraints_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.verify("alice = green", solution={"variables": VARIABLES})
        self.assertIn("without 'constraints'", str(ctx.exception))

    def test_an_unknown_constraint_kind_is_an_error_not_an_ignored_constraint(self) -> None:
        solution = {"variables": VARIABLES, "constraints": [{"kind": "riddle"}]}
        with self.assertRaises(ValueError) as ctx:
            self.verify("alice = green", solution=solution)
        self.assertIn("not one of", str(ctx.exception))

    def test_a_constraint_missing_its_operands_is_an_error(self) -> None:
        for constraint in ({"kind": "equals", "value": "blue"}, {"kind": "equals", "var": "bob"}):
            with self.subTest(constraint=constraint):
                with self.assertRaises(ValueError):
                    constraint_failure(constraint, {"bob": "blue"}, VARIABLES)

    def test_the_reward_term_is_named(self) -> None:
        outcome = self.verify("alice = green\nbob = blue\ncarol = red")
        self.assertEqual(outcome.as_terms("rule"), {"rule": 1.0})


class FakeSandbox:
    """A sandbox that answers from a script, so the verifier's branches are testable here."""

    name = "fake sandbox"

    def __init__(self, exit_code: int | None = 0, timed_out: bool = False,
                 output: str = "", error: Exception | None = None) -> None:
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.output = output
        self.error = error
        self.seen: dict[str, object] = {}

    def run(self, command: str, workdir: Path, timeout: float) -> SandboxResult:
        attempt = workdir / "main.py"
        self.seen = {
            "command": command,
            "workdir": workdir,
            "timeout": timeout,
            "attempt": attempt.read_text(encoding="utf-8") if attempt.exists() else None,
            "entries": sorted(entry.name for entry in workdir.iterdir()),
        }
        if self.error is not None:
            raise self.error
        return SandboxResult(self.exit_code, self.timed_out, self.output, self.name)


class UnitTestVerifierTests(unittest.TestCase):
    TASK = {"id": "u", "tests": "python3 -m pytest -q"}

    def verify(self, sandbox: FakeSandbox, response: str = "def add(a, b):\n    return a + b",
               **fields: object):
        task = {**self.TASK, **fields}
        return UnitTestVerifier(sandbox).verify(task, response)

    def test_a_passing_run_scores_one_and_names_the_sandbox(self) -> None:
        outcome = self.verify(FakeSandbox(exit_code=0))
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)
        self.assertIn("fake sandbox", outcome.detail)
        self.assertIn("tests passed", outcome.detail)

    def test_the_attempt_is_written_where_the_tests_expect_it(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox, attempt_file="solution.py", response="x = 1")
        self.assertEqual(sandbox.seen["entries"], ["solution.py"])

    def test_the_default_attempt_file_is_used_when_the_task_names_none(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox, response="x = 1\n")
        self.assertEqual(sandbox.seen["entries"], [DEFAULT_ATTEMPT_FILE])
        self.assertEqual(sandbox.seen["attempt"], "x = 1\n")

    def test_the_task_s_tests_command_is_what_runs(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox)
        self.assertEqual(sandbox.seen["command"], self.TASK["tests"])

    def test_the_timeout_is_the_task_s_or_the_default(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox)
        self.assertEqual(sandbox.seen["timeout"], DEFAULT_TEST_TIMEOUT_S)
        self.verify(sandbox, test_timeout=3)
        self.assertEqual(sandbox.seen["timeout"], 3.0)

    def test_a_failing_run_scores_zero_and_quotes_the_output(self) -> None:
        sandbox = FakeSandbox(exit_code=1, output="line one\nAssertionError: 5 != -1\n")
        outcome = self.verify(sandbox)
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("exit status 1", outcome.detail)
        self.assertIn("AssertionError: 5 != -1", outcome.detail)

    def test_a_timeout_is_a_failure_that_says_timed_out(self) -> None:
        # The book's rule: an environment that hangs must not look like a slow model.
        outcome = self.verify(FakeSandbox(exit_code=None, timed_out=True))
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("timed out", outcome.detail)
        self.assertIn("failed attempt, not an error", outcome.detail)

    def test_an_unavailable_sandbox_raises_rather_than_scoring_zero(self) -> None:
        # 'This machine cannot run the tests' is not the model's fault, and a 0.0 would be.
        sandbox = FakeSandbox(error=SandboxUnavailable("no container runtime here"))
        with self.assertRaises(SandboxUnavailable):
            self.verify(sandbox)

    def test_an_empty_response_fails_without_starting_a_sandbox(self) -> None:
        sandbox = FakeSandbox()
        outcome = self.verify(sandbox, response="   \n")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)
        self.assertIn("nothing to run the tests against", outcome.detail)
        self.assertEqual(sandbox.seen, {})

    def test_a_fenced_attempt_is_unwrapped_before_it_is_written(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox, response="```python\nx = 1\n```")
        self.assertEqual(sandbox.seen["attempt"], "x = 1\n")

    def test_the_working_directory_is_gone_afterwards(self) -> None:
        sandbox = FakeSandbox()
        self.verify(sandbox)
        self.assertFalse(Path(str(sandbox.seen["workdir"])).exists())

    def test_can_verify_needs_a_non_empty_tests_command(self) -> None:
        verifier = UnitTestVerifier(FakeSandbox())
        self.assertTrue(verifier.can_verify({"tests": "true"}))
        self.assertFalse(verifier.can_verify({"tests": "   "}))
        self.assertFalse(verifier.can_verify({"tests": 5}))
        self.assertFalse(verifier.can_verify({}))

    def test_an_attempt_file_that_is_a_path_is_an_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.verify(FakeSandbox(), attempt_file="../../etc/passwd")
        self.assertIn("not a plain file name", str(ctx.exception))

    def test_a_non_positive_timeout_is_an_error(self) -> None:
        # A zero is not "absent": the first version resolved the timeout with `or`, which
        # quietly replaced 0 with the 20-second default. The test caught it; resolution is
        # explicit about None now.
        for bad in (0, -5):
            with self.subTest(test_timeout=bad):
                with self.assertRaises(ValueError) as ctx:
                    self.verify(FakeSandbox(), test_timeout=bad)
                self.assertIn("non-positive", str(ctx.exception))

    def test_the_reward_term_is_named(self) -> None:
        self.assertEqual(self.verify(FakeSandbox()).as_terms("rule"), {"rule": 1.0})


class LocalSandboxTests(unittest.TestCase):
    sandbox = LocalSandbox(why="this test wrote the command")

    def test_a_successful_command_reports_status_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.sandbox.run("printf 'ok'", Path(tmp), timeout=10)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)
        self.assertIn("ok", result.output)
        self.assertEqual(result.sandbox, self.sandbox.name)

    def test_it_runs_in_the_working_directory_it_was_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "attempt.py").write_text("print('hello')\n", encoding="utf-8")
            result = self.sandbox.run("python3 attempt.py", Path(tmp), timeout=30)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.output)

    def test_a_failing_command_reports_its_status_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.sandbox.run("echo nope; exit 3", Path(tmp), timeout=10)
        self.assertEqual(result.exit_code, 3)
        self.assertIn("nope", result.output)
        self.assertEqual(result.tail(), "nope")

    def test_a_hanging_command_is_reported_as_a_timeout_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.sandbox.run("sleep 30", Path(tmp), timeout=0.5)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

    def test_it_refuses_to_run_without_a_reason(self) -> None:
        with self.assertRaises(ValueError):
            LocalSandbox(why="  ").run("true", Path(tempfile.gettempdir()), timeout=5)


class DockerSandboxTests(unittest.TestCase):
    def test_the_command_line_has_no_network_and_a_name(self) -> None:
        argv = DockerSandbox().argv("pytest -q", Path("/tmp/work"), "p1-unit-abc")
        self.assertEqual(argv[:2], ["docker", "run"])
        self.assertIn("--network=none", argv)
        self.assertIn("--name", argv)
        self.assertIn("p1-unit-abc", argv)
        self.assertIn("--memory=512m", argv)
        self.assertIn("/tmp/work:/work", argv)
        self.assertEqual(argv[-3:], ["sh", "-lc", "pytest -q"])

    def test_the_container_runs_as_the_calling_user(self) -> None:
        # Measured live: running as root, `import main` writes __pycache__ into the bind mount,
        # and the host user cannot delete it - so cleanup raised PermissionError and the run died
        # with a traceback instead of a score.
        argv = DockerSandbox().argv("true", Path("/tmp"), "c")
        self.assertEqual(argv[argv.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")

    def test_an_explicit_user_wins_over_the_caller(self) -> None:
        argv = DockerSandbox(user="1000:1000").argv("true", Path("/tmp"), "c")
        self.assertEqual(argv[argv.index("--user") + 1], "1000:1000")

    def test_the_image_is_configurable(self) -> None:
        self.assertIn("python:3.13-alpine", DockerSandbox().argv("true", Path("/tmp"), "c"))
        self.assertIn("pytorch/pytorch:latest",
                      DockerSandbox(image="pytorch/pytorch:latest").argv("true", Path("/tmp"), "c"))

    def test_a_missing_docker_is_unavailable_not_a_zero(self) -> None:
        # Patched rather than skipped, so the assertion holds on machines that do have docker.
        original = shutil.which
        shutil.which = lambda name: None  # type: ignore[assignment]
        try:
            with self.assertRaises(SandboxUnavailable) as ctx:
                DockerSandbox().run("true", Path(tempfile.gettempdir()), timeout=5)
        finally:
            shutil.which = original  # type: ignore[assignment]
        self.assertIn("docker is not on PATH", str(ctx.exception))


class TaskLoadingTests(unittest.TestCase):
    def write(self, lines: list[str]) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "tasks.jsonl"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp

    def test_fields_are_kept_alongside_the_known_keys(self) -> None:
        task = Task.from_dict({"id": "a", "prompt": "p", "verifier": "v", "tolerance": 0.1})
        self.assertEqual(task.fields, {"tolerance": 0.1})
        self.assertEqual(task.as_mapping()["tolerance"], 0.1)

    def test_a_row_missing_a_required_key_is_rejected(self) -> None:
        for missing in ("id", "prompt", "verifier"):
            row = {"id": "a", "prompt": "p", "verifier": "v"}
            row.pop(missing)
            with self.assertRaises(TaskSetError):
                Task.from_dict(row)

    def test_blank_lines_are_skipped(self) -> None:
        path = self.write(['{"id": "a", "prompt": "p", "verifier": "v"}', "", ""])
        self.assertEqual(len(load_tasks(path)), 1)

    def test_duplicate_ids_are_rejected_by_line_number(self) -> None:
        path = self.write(['{"id": "a", "prompt": "p", "verifier": "v"}',
                           '{"id": "a", "prompt": "q", "verifier": "v"}'])
        with self.assertRaises(TaskSetError) as ctx:
            load_tasks(path)
        self.assertIn("duplicate task id", str(ctx.exception))

    def test_a_malformed_row_is_reported_with_its_line(self) -> None:
        path = self.write(['{"id": "a", "prompt": "p", "verifier": "v"}', "{not json}"])
        with self.assertRaises(TaskSetError) as ctx:
            load_tasks(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_the_real_fixture_loads_with_the_documented_shape(self) -> None:
        tasks = load_tasks(FIXTURES / "tasks.jsonl")
        self.assertEqual(len(tasks), 21)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "exact_match"), 2)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "numeric"), 15)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "unit_test"), 2)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "logic_puzzle"), 2)


class ResponderTests(unittest.TestCase):
    def test_answers_cycle_by_sample_index(self) -> None:
        responder = FixtureResponder({"t": ["first", "second"]})
        task = Task(id="t", prompt="p", verifier="exact_match")
        self.assertEqual(responder.respond(task, 0), "first")
        self.assertEqual(responder.respond(task, 1), "second")
        self.assertEqual(responder.respond(task, 2), "first")

    def test_a_missing_answer_is_an_error_not_an_empty_string(self) -> None:
        with self.assertRaises(TaskSetError):
            FixtureResponder({"a": ["x"]}).respond(Task(id="b", prompt="p", verifier="v"), 0)

    def test_an_empty_responder_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FixtureResponder({})

    def test_loading_from_jsonl_groups_rows_by_prompt(self) -> None:
        responder = FixtureResponder.from_jsonl(FIXTURES / "responses.jsonl")
        self.assertEqual(len(responder._answers["t01"]), 2)  # noqa: SLF001 - test of the loader


class StubContractTests(unittest.TestCase):
    """The rule a stub has to keep, asserted for whatever ``UNIMPLEMENTED`` currently holds."""

    PROBE = {"id": "probe", "prompt": "irrelevant", "answer": "1", "tolerance": 0.1,
             "tests": "true", "solution": {"variables": ["a"], "constraints": [{"kind": "distinct"}]}}

    def test_every_unimplemented_name_declines_and_raises(self) -> None:
        registry = default_registry()
        for name in UNIMPLEMENTED:
            with self.subTest(name=name):
                verifier = registry.get(name)
                self.assertFalse(verifier.can_verify(self.PROBE))
                with self.assertRaises(NotImplementedError):
                    verifier.verify(self.PROBE, "anything")

    def test_the_suite_is_complete(self) -> None:
        # The current state, asserted rather than assumed: four declared verifiers, none of
        # them a stub. A new stub has to arrive with the checks that describe it.
        self.assertEqual(UNIMPLEMENTED, ())
        self.assertEqual(default_registry().names,
                         ["exact_match", "logic_puzzle", "numeric", "unit_test"])


class CoveragePartitionTests(unittest.TestCase):
    def test_every_task_in_the_fixture_is_now_covered(self) -> None:
        tasks = load_tasks(FIXTURES / "tasks.jsonl")
        covered, uncovered = partition_by_coverage(tasks, default_registry())
        self.assertEqual(len(covered), 21)
        self.assertEqual(uncovered, [])
        self.assertEqual([task.id for task in covered if task.verifier == "numeric"],
                         ["t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09", "t10", "t11",
                          "e01", "e02", "m02", "m03", "m04"])
        self.assertEqual([task.id for task in covered if task.verifier == "unit_test"], ["t12", "e05"])
        self.assertEqual([task.id for task in covered if task.verifier == "logic_puzzle"],
                         ["t13", "e04"])

    def test_a_task_that_declares_a_verifier_which_declines_it_is_reported_with_a_reason(self) -> None:
        # A logic_puzzle task with no solution: the verifier is registered, but this task is
        # not something it can judge, and coverage has to say so rather than count it.
        task = Task(id="t", prompt="p", verifier="logic_puzzle")
        covered, uncovered = partition_by_coverage([task], default_registry())
        self.assertEqual(covered, [])
        self.assertEqual(len(uncovered), 1)
        self.assertIn("cannot verify this task as written", uncovered[0][1])
        self.assertIn("logic_puzzle", uncovered[0][1])

    def test_a_unit_test_task_without_tests_is_declined(self) -> None:
        task = Task(id="t", prompt="p", verifier="unit_test")
        covered, uncovered = partition_by_coverage([task], default_registry())
        self.assertEqual(covered, [])
        self.assertIn("unit_test", uncovered[0][1])

    def test_a_task_naming_an_unknown_verifier_is_uncovered(self) -> None:
        tasks = [Task(id="x", prompt="p", verifier="ghost")]
        covered, uncovered = partition_by_coverage(tasks, default_registry())
        self.assertEqual(covered, [])
        self.assertIn("no verifier named", uncovered[0][1])


class EpisodeTests(unittest.TestCase):
    def episode(self, task: Task, response: str):
        return run_episode(task, response, default_registry(), 0, Provenance())

    def test_a_row_is_complete_and_self_consistent(self) -> None:
        row = self.episode(Task(id="t", prompt="What is 2+2?", verifier="exact_match",
                                fields={"answer": "4"}), "2+2 = 4\n\n4")
        self.assertEqual(row.episode_id, "t-s0")
        self.assertEqual(row.turn_count, 1)          # single-turn: a model segment, no tool
        self.assertIsNotNone(row.mask)
        self.assertEqual(sum(row.mask or []), row.model_tokens)
        self.assertTrue(row.extras["passed"])
        self.assertEqual(row.reward_terms, {"rule": 1.0})

    def test_a_numeric_task_is_scored_by_its_tolerance(self) -> None:
        task = Task(id="t", prompt="pi?", verifier="numeric",
                    fields={"answer": "3.14", "tolerance": 0.01})
        row = self.episode(task, "pi is 3.14159...\n\n3.14")
        self.assertEqual(row.extras["verifier"], "numeric")
        self.assertTrue(row.extras["passed"])

    def test_a_logic_puzzle_is_scored_by_its_constraints(self) -> None:
        task = Task(id="t", prompt="who chose what?", verifier="logic_puzzle",
                    fields={"solution": LogicPuzzleVerifierTests.SOLUTION})
        row = self.episode(task, "alice = green\nbob = blue\ncarol = red")
        self.assertEqual(row.extras["verifier"], "logic_puzzle")
        self.assertTrue(row.extras["passed"])

    def test_a_declined_task_raises_rather_than_scoring_zero(self) -> None:
        for task in (
            # a numeric task with no tolerance: missing the field this verifier needs
            Task(id="t", prompt="pi?", verifier="numeric", fields={"answer": "3.14"}),
            # a unit-test task with no tests: nothing to run
            Task(id="t", prompt="p", verifier="unit_test"),
            # a logic-puzzle task with no solution: nothing to check against
            Task(id="t", prompt="p", verifier="logic_puzzle"),
        ):
            with self.subTest(verifier=task.verifier, fields=task.fields):
                with self.assertRaises(MissingVerifier):
                    self.episode(task, "3.14")

    def test_provenance_travels_with_the_row(self) -> None:
        row = run_episode(
            Task(id="t", prompt="p", verifier="exact_match", fields={"answer": "1"}),
            "1", default_registry(), 0,
            Provenance(env_version="test-env", config_hash="sha256:cfg", code_sha="git:abc"),
        )
        self.assertEqual((row.env_version, row.config_hash, row.code_sha),
                         ("test-env", "sha256:cfg", "git:abc"))


class TaskSetRunTests(unittest.TestCase):
    def rows(self):
        return fixture_rows()

    def test_two_samples_per_task_form_groups(self) -> None:
        rows = self.rows()
        self.assertEqual(len(rows), 42)
        grouped: dict[str, set[int]] = {}
        for row in rows:
            grouped.setdefault(row.group_id, set()).add(row.sample_index)
        self.assertEqual(len(grouped), 21)
        self.assertTrue(all(indices == {0, 1} for indices in grouped.values()))

    def test_every_row_has_a_usable_mask_and_a_recorded_turn_count(self) -> None:
        for row in self.rows():
            self.assertTrue(row.is_trainable(), row.episode_id)
            self.assertEqual(row.turn_count, 1)

    def test_the_fixture_pass_rate_is_what_the_check_script_asserts(self) -> None:
        self.assertEqual(pass_rate(self.rows()), (21, 42))

    def test_samples_per_task_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            run_taskset([], FixtureResponder({"a": ["x"]}), default_registry(), samples_per_task=0)

    def test_the_report_names_the_misses(self) -> None:
        text = report(self.rows())
        self.assertIn("21/42 (50%)", text)
        self.assertIn("missed:", text)
        self.assertIn("tokens:", text)

    def test_an_empty_run_reports_rather_than_crashing(self) -> None:
        self.assertEqual(report([]), "no rollouts")


class FixtureContractTests(unittest.TestCase):
    """The fixtures are inputs to a claim, so their shape is asserted, not assumed."""

    @staticmethod
    def counts() -> dict[str, int]:
        counted: dict[str, int] = {}
        for line in (FIXTURES / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = json.loads(line)["prompt_id"]
                counted[key] = counted.get(key, 0) + 1
        return counted

    def test_responses_cover_every_scoreable_task_twice(self) -> None:
        counted = self.counts()
        tasks = load_tasks(FIXTURES / "tasks.jsonl")
        covered, _ = partition_by_coverage(tasks, default_registry())
        for task in covered:
            self.assertEqual(counted.get(task.id), 2, f"{task.id} has the wrong number of responses")

    def test_no_response_exists_for_a_task_nothing_can_score(self) -> None:
        counted = self.counts()
        self.assertEqual(set(counted), {task.id for task in load_tasks(FIXTURES / "tasks.jsonl")})

    def test_the_numeric_task_is_one_pass_and_one_fail(self) -> None:
        # That pair is what makes t11's group mixed: inside the tolerance, then outside it.
        rows = [row for row in fixture_rows() if row.prompt_id == "t11"]
        self.assertEqual([row.extras["passed"] for row in rows], [True, False])
        self.assertEqual({row.reward() for row in rows}, {0.0, 1.0})

    def test_the_unit_test_task_is_one_pass_and_one_fail(self) -> None:
        # t12's pair is the sandbox's two outcomes: the tests exit 0, then they do not.
        rows = [row for row in fixture_rows() if row.prompt_id == "t12"]
        self.assertEqual([row.extras["passed"] for row in rows], [True, False])
        self.assertEqual({row.reward() for row in rows}, {0.0, 1.0})

    def test_the_logic_puzzle_task_is_one_pass_and_one_near_miss(self) -> None:
        # t13's pair: a consistent assignment, then one that breaks a single constraint - and
        # the near miss has to read as one in the record, not as a bare zero.
        rows = [row for row in fixture_rows() if row.prompt_id == "t13"]
        self.assertEqual([row.extras["passed"] for row in rows], [True, False])
        self.assertIn("all 3 constraints hold", rows[0].extras["detail"])
        self.assertIn("satisfied 2 of 3 constraints", rows[1].extras["detail"])

    def test_every_fixture_row_says_which_sandbox_scored_it(self) -> None:
        # A row scored without a container must be distinguishable from one scored inside one.
        for row in fixture_rows():
            if row.prompt_id in ("t12", "e05"):
                self.assertIn("local subprocess (no isolation)", row.extras["detail"])

    def test_the_added_tasks_are_one_pass_and_one_fail_each(self) -> None:
        # The five tasks added for difficulty: a pair that does not move adds no signal.
        for task_id in ("e01", "e02", "e03", "e04", "e05", "m02", "m03", "m04"):
            with self.subTest(task_id=task_id):
                rows = [row for row in fixture_rows() if row.prompt_id == task_id]
                self.assertEqual([row.extras["passed"] for row in rows], [True, False])
                self.assertEqual({row.reward() for row in rows}, {0.0, 1.0})

    def test_the_prose_attempt_scores_zero_for_its_form_not_its_arithmetic(self) -> None:
        # The live failure, pinned in the offline fixture: the arithmetic is right (both 3 and
        # 36 are stated) and the attempt scores zero because no answer was declared. A fixture
        # of tidy answers only could never catch this.
        rows = [row for row in fixture_rows() if row.prompt_id == "e02"]
        self.assertEqual([row.extras["passed"] for row in rows], [True, False])
        self.assertTrue(str(rows[0].extras["detail"]).startswith(f"{FORM_PREFIX}ok"))
        self.assertTrue(str(rows[1].extras["detail"]).startswith(f"{FORM_PREFIX}no"))
        self.assertIn("the expected value is also in that line", str(rows[1].extras["detail"]))

    def test_the_fixture_keeps_two_flat_groups_on_purpose(self) -> None:
        # Not every group can be made to move: this responder is reliable on t02 and out of its
        # depth on t04. A task set that claims every group has spread is claiming too much.
        flat_high = [row for row in fixture_rows() if row.prompt_id == "t02"]
        flat_low = [row for row in fixture_rows() if row.prompt_id == "t04"]
        self.assertEqual([row.extras["passed"] for row in flat_high], [True, True])
        self.assertEqual([row.extras["passed"] for row in flat_low], [False, False])


if __name__ == "__main__":
    unittest.main()
