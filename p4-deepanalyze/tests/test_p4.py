"""Tests for the P4 loop: the protocol, the verdict, the tasks, and the episode itself.

The sandbox is replaced by ``FakeSandbox`` here.  Not for speed alone - the real one needs a
container runtime, and a test that needs Docker is a test that will be skipped exactly on the
machine where it matters.  What is tested is the part that decides what a result *means*:
which replies are refused, how a timeout is separated from a wrong answer, and whether an
episode that never reported can ever come out solved.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from p4_deepanalyze import protocol, verify  # noqa: E402
from p4_deepanalyze.agent import run_episode  # noqa: E402
from p4_deepanalyze.client import ScriptedClient  # noqa: E402
from p4_deepanalyze.sandbox import LocalSandbox, Observation  # noqa: E402
from p4_deepanalyze.tasks import Task, load  # noqa: E402

TASK = Task(
    id="t-one",
    question="What is the sum of the values?",
    table="k,v\na,1\nb,2",
    expected="3",
)


class FakeSandbox:
    """Records what it was asked to run, and answers with what the test decided."""

    def __init__(self, observations=None):
        self.observations = list(observations or [])
        self.calls: list[tuple[str, dict]] = []

    def run(self, code: str, files=None):
        self.calls.append((code, files or {}))
        if self.observations:
            return self.observations.pop(0)
        return Observation(exit_code=0, stdout="3\n", stderr="", timed_out=False, duration_ms=5)


def _clean() -> Observation:
    return Observation(exit_code=0, stdout="3\n", stderr="", timed_out=False, duration_ms=5)


class ProtocolTests(unittest.TestCase):
    def test_code_and_report_are_parsed(self):
        self.assertEqual(protocol.parse("<code>print(1)</code>").kind, "code")
        self.assertEqual(protocol.parse("<report>42</report>").text, "42")

    def test_analyze_is_kept_but_not_acted_on(self):
        action = protocol.parse("<analyze>sum it</analyze>\n<report>3</report>")
        self.assertEqual(action.kind, "report")
        self.assertEqual(action.analyzed, "sum it")

    def test_prose_is_a_protocol_error_with_a_reason(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse("I think the answer is 3.")
        self.assertEqual(caught.exception.reason, "no marker")

    def test_two_actions_at_once_is_refused(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse("<code>print(1)</code><report>1</report>")
        self.assertEqual(caught.exception.reason, "both code and report")

    def test_an_unclosed_marker_is_named_as_such(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse("<code>print(1)")
        self.assertEqual(caught.exception.reason, "unclosed marker")

    def test_empty_reply_is_refused(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse("   ")

    def test_silent_success_is_flagged_and_explained(self):
        warnings = protocol.observation_warnings(0, "", "", False)
        self.assertEqual(warnings, ["no_output"])
        text = protocol.format_observation(0, "", "", False, warnings)
        self.assertIn("printed nothing", text)
        self.assertIn("print(...)", text)

    def test_multiline_output_is_flagged_with_the_count(self):
        warnings = protocol.observation_warnings(0, "0   -80\n2   -25\n3   -35\n", "", False)
        self.assertEqual(warnings, ["multiline_output"])
        text = protocol.format_observation(0, "a\nb\nc", "", False, warnings)
        self.assertIn("printed 3 lines", text)
        self.assertIn("single value", text)

    def test_the_prompt_says_each_program_is_a_new_process(self):
        self.assertIn("variables from an earlier turn do not exist", protocol.SYSTEM_PROMPT)
        self.assertIn("Print exactly one value", protocol.SYSTEM_PROMPT)

    def test_a_clean_single_value_run_has_no_warnings(self):
        self.assertEqual(protocol.observation_warnings(0, "1740.0\n", "", False), [])
        self.assertEqual(protocol.observation_warnings(1, "", "KeyError", False), [])

    def test_observation_text_separates_a_timeout(self):
        timed_out = protocol.format_observation(None, "", "", True)
        failed = protocol.format_observation(1, "", "Traceback", False)
        self.assertIn("ran longer than the sandbox timeout", timed_out)
        self.assertIn("exit=1", failed)
        self.assertIn("Traceback", failed)


class VerifyTests(unittest.TestCase):
    def test_the_last_number_is_the_answer(self):
        self.assertTrue(verify.is_correct("The total is 3", "3"))
        self.assertFalse(verify.is_correct("3 minus 1 is 2", "3"))

    def test_tolerance_and_text(self):
        self.assertTrue(verify.is_correct("0.667", "0.6667", tolerance=0.01))
        self.assertFalse(verify.is_correct("0.5", "0.6667", tolerance=0.01))
        self.assertTrue(verify.is_correct("north", "North"))

    def test_no_report_is_not_correct(self):
        self.assertFalse(verify.is_correct(None, "3"))

    def test_why_explains_a_miss(self):
        self.assertEqual(verify.why("7", "3"), "reported 7, expected 3")


class TaskTests(unittest.TestCase):
    def test_fixture_file_loads(self):
        tasks = load(HERE.parent / "fixtures" / "tasks.jsonl")
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len({task.id for task in tasks}), 6)
        for task in tasks:
            self.assertIn("table.csv", task.files())
            self.assertTrue(task.preview().startswith(task.columns()[0]))

    def test_duplicate_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            row = {"id": "dup", "question": "q", "table": "a\n1", "expected": "1"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)

    def test_a_table_without_rows_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps({"id": "x", "question": "q", "table": "a", "expected": "1"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)


class EpisodeTests(unittest.TestCase):
    def test_code_then_report_solves(self):
        client = ScriptedClient(["<code>print(3)</code>", "<report>3</report>"])
        sandbox = FakeSandbox()
        episode = run_episode(TASK, client, sandbox, max_turns=3)
        self.assertTrue(episode.solved)
        self.assertEqual(len(sandbox.calls), 1)
        self.assertIn("table.csv", sandbox.calls[0][1])
        self.assertEqual(episode.turns[0].kind, "code")
        self.assertEqual(episode.turns[1].kind, "report")

    def test_the_observation_reaches_the_model(self):
        client = ScriptedClient(["<code>print(3)</code>", "<report>3</report>"])
        run_episode(TASK, client, FakeSandbox(), max_turns=3)
        second_call = client.seen[1]
        self.assertEqual(second_call[-1]["role"], "user")
        self.assertIn("<observation exit=0>", second_call[-1]["content"])

    def test_a_timeout_is_counted_and_not_scored(self):
        timed = Observation(exit_code=None, stdout="", stderr="", timed_out=True, duration_ms=20000)
        client = ScriptedClient(["<code>while True: pass</code>", "<report>3</report>"])
        episode = run_episode(TASK, client, FakeSandbox([timed]), max_turns=3)
        self.assertEqual(episode.timeouts, 1)
        self.assertEqual(episode.sandbox_errors, 0)
        self.assertTrue(episode.solved)  # the model still reported, from its own reasoning

    def test_a_crash_is_a_sandbox_error(self):
        crashed = Observation(exit_code=1, stdout="", stderr="KeyError", timed_out=False, duration_ms=9)
        client = ScriptedClient(["<code>1/0</code>", "<report>3</report>"])
        episode = run_episode(TASK, client, FakeSandbox([crashed]), max_turns=3)
        self.assertEqual(episode.sandbox_errors, 1)

    def test_a_silent_program_is_recorded_as_a_lost_turn(self):
        silent = Observation(exit_code=0, stdout="", stderr="", timed_out=False, duration_ms=7)
        client = ScriptedClient(["<code>max_month</code>", "<report>March</report>", "<report>3</report>"])
        episode = run_episode(TASK, client, FakeSandbox([silent]), max_turns=3)
        self.assertEqual(episode.lost_turns, 1)
        self.assertEqual(episode.forced_repairs, 1)
        self.assertEqual(episode.turns[0].warnings, ["no_output"])
        self.assertIn("printed nothing", client.seen[1][-1]["content"])

    def test_a_report_after_a_silent_run_is_refused_once(self):
        silent = Observation(exit_code=0, stdout="", stderr="", timed_out=False, duration_ms=7)
        client = ScriptedClient(
            ["<code>max_month</code>", "<report>March</report>", "<code>print(3)</code>", "<report>3</report>"]
        )
        episode = run_episode(TASK, client, FakeSandbox([silent, _clean()]), max_turns=4)
        self.assertEqual(episode.forced_repairs, 1)
        self.assertEqual(episode.turns[1].kind, "repair-request")
        self.assertEqual(episode.turns[2].kind, "code")
        self.assertEqual(len(episode.turns), 4)
        self.assertTrue(episode.solved)

    def test_the_second_report_is_accepted(self):
        silent = Observation(exit_code=0, stdout="", stderr="", timed_out=False, duration_ms=7)
        client = ScriptedClient(["<code>max_month</code>", "<report>March</report>", "<report>3</report>"])
        episode = run_episode(TASK, client, FakeSandbox([silent]), max_turns=4)
        self.assertEqual(episode.forced_repairs, 1)
        self.assertTrue(episode.solved)  # 3 is the task's answer; the harness does not insist twice

    def test_prose_stops_the_episode_as_a_protocol_error(self):
        client = ScriptedClient(["The answer is 3."])
        episode = run_episode(TASK, client, FakeSandbox(), max_turns=3)
        self.assertFalse(episode.solved)
        self.assertEqual(episode.protocol_errors, 1)
        self.assertIsNone(episode.report)

    def test_no_report_within_the_limit_is_unsolved(self):
        client = ScriptedClient(["<code>print(1)</code>"] * 2)
        episode = run_episode(TASK, client, FakeSandbox(), max_turns=2)
        self.assertFalse(episode.solved)
        self.assertIsNone(episode.report)
        self.assertIn("no report within 2 turns", episode.reason)

    def test_a_wrong_report_is_unsolved_with_a_reason(self):
        client = ScriptedClient(["<report>9</report>"])
        episode = run_episode(TASK, client, FakeSandbox(), max_turns=2)
        self.assertFalse(episode.solved)
        self.assertEqual(episode.reason, "reported 9, expected 3")


class LocalSandboxTests(unittest.TestCase):
    def test_it_runs_code_and_sees_the_table(self):
        sandbox = LocalSandbox(timeout_s=10)
        observation = sandbox.run(
            "import csv\n"
            "with open('table.csv') as handle:\n"
            "    print(sum(int(row['v']) for row in csv.DictReader(handle)))\n",
            TASK.files(),
        )
        self.assertEqual(observation.exit_code, 0)
        self.assertEqual(observation.stdout.strip(), "3")

    def test_it_reports_a_timeout(self):
        sandbox = LocalSandbox(timeout_s=0.5)
        observation = sandbox.run("import time\ntime.sleep(5)")
        self.assertTrue(observation.timed_out)
        self.assertIsNone(observation.exit_code)

    def test_a_failing_program_keeps_its_exit_code(self):
        observation = LocalSandbox(timeout_s=10).run("raise SystemExit(3)")
        self.assertEqual(observation.exit_code, 3)
        self.assertFalse(observation.timed_out)


if __name__ == "__main__":
    unittest.main()
