"""Tests for P1: extraction, verification, task loading, and the loop.

    python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests

The directory is not a valid Python package name (it has a hyphen), so its own path is
the discovery root and this file puts the repository and ``p1-verifiers/src`` on
``sys.path`` itself.
"""

from __future__ import annotations

import json
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
    DockerSandbox,
    ExactMatchVerifier,
    FixtureResponder,
    LocalSandbox,
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
from p1_verifiers.verifiers import final_answer, last_number  # noqa: E402

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


class LastNumberTests(unittest.TestCase):
    def test_the_last_number_wins_not_the_first(self) -> None:
        self.assertEqual(last_number("pi is 3.14159, so 3.14"), 3.14)

    def test_integers_decimals_and_signs(self) -> None:
        self.assertEqual(last_number("the answer is 12"), 12.0)
        self.assertEqual(last_number("x = -20"), -20.0)
        self.assertEqual(last_number("+7"), 7.0)

    def test_thousands_separators_and_exponents_parse(self) -> None:
        self.assertEqual(last_number("a population of 1,000,000"), 1000000.0)
        self.assertEqual(last_number("about 1e3 tokens"), 1000.0)

    def test_a_response_without_a_number_has_no_number(self) -> None:
        self.assertIsNone(last_number("I do not know."))
        self.assertIsNone(last_number(""))

    def test_a_date_reads_as_its_last_numeric_fragment(self) -> None:
        # Documented blind spot, not an accident: this helper is for arithmetic answers.
        self.assertEqual(last_number("2026-10-10"), -10.0)


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
        for response in ("3.14", "3.140", "π is 3.14159, so 3.14", "the value is 3.14 exactly"):
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
        self.assertEqual(len(tasks), 12)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "exact_match"), 10)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "numeric"), 1)
        self.assertEqual(sum(1 for task in tasks if task.verifier == "unit_test"), 1)


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


class CoveragePartitionTests(unittest.TestCase):
    def test_every_task_in_the_fixture_is_now_covered(self) -> None:
        tasks = load_tasks(FIXTURES / "tasks.jsonl")
        covered, uncovered = partition_by_coverage(tasks, default_registry())
        self.assertEqual(len(covered), 12)
        self.assertEqual(uncovered, [])
        self.assertEqual([task.id for task in covered if task.verifier == "numeric"], ["t11"])
        self.assertEqual([task.id for task in covered if task.verifier == "unit_test"], ["t12"])

    def test_a_task_whose_verifier_is_still_a_stub_is_declined_with_a_reason(self) -> None:
        task = Task(id="t", prompt="p", verifier="logic_puzzle", fields={"solution": {}})
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

    def test_a_declined_task_raises_rather_than_scoring_zero(self) -> None:
        for task in (
            # a numeric task with no tolerance: missing the field this verifier needs
            Task(id="t", prompt="pi?", verifier="numeric", fields={"answer": "3.14"}),
            # a unit-test task with no tests: nothing to run
            Task(id="t", prompt="p", verifier="unit_test"),
            # a verifier that is still a stub
            Task(id="t", prompt="p", verifier="logic_puzzle", fields={"solution": {}}),
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
        self.assertEqual(len(rows), 24)
        grouped: dict[str, set[int]] = {}
        for row in rows:
            grouped.setdefault(row.group_id, set()).add(row.sample_index)
        self.assertEqual(len(grouped), 12)
        self.assertTrue(all(indices == {0, 1} for indices in grouped.values()))

    def test_every_row_has_a_usable_mask_and_a_recorded_turn_count(self) -> None:
        for row in self.rows():
            self.assertTrue(row.is_trainable(), row.episode_id)
            self.assertEqual(row.turn_count, 1)

    def test_the_fixture_pass_rate_is_what_the_check_script_asserts(self) -> None:
        self.assertEqual(pass_rate(self.rows()), (16, 24))

    def test_samples_per_task_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            run_taskset([], FixtureResponder({"a": ["x"]}), default_registry(), samples_per_task=0)

    def test_the_report_names_the_misses(self) -> None:
        text = report(self.rows())
        self.assertIn("16/24 (67%)", text)
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

    def test_every_fixture_row_says_which_sandbox_scored_it(self) -> None:
        # A row scored without a container must be distinguishable from one scored inside one.
        for row in fixture_rows():
            if row.prompt_id == "t12":
                self.assertIn("local subprocess (no isolation)", row.extras["detail"])


if __name__ == "__main__":
    unittest.main()
