"""The P5 gate: the world is solvable, the protocol refuses what it should, and the accounting adds up.

Everything here runs without a GPU, a server or a model, which is the same reason the other projects'
checks do: a test that needs the machine you are least likely to have is a test that never runs.

    python3 p5-openclaw-rl/checks/check_p5.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))

from p5_openclaw import credit, protocol  # noqa: E402
from p5_openclaw.world import DEMO_TASKS, EXTRA_TASKS, TASKS, World, oracle_episode, solved  # noqa: E402

DEMOS = HERE.parent / "fixtures" / "demonstrations.jsonl"


class WorldTests(unittest.TestCase):
    def test_every_task_is_solvable(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                world = oracle_episode(task)
                self.assertTrue(solved(world), f"{task.id}: oracle submitted {world.submitted!r}")
                self.assertEqual(world.errors, 0)

    def test_the_tasks_are_not_all_the_same_answer(self):
        # A task set where every answer is "North" would let a lazy policy look fine.
        answers = {task.answer for task in TASKS}
        self.assertGreaterEqual(len(answers), 4)

    def test_a_tool_error_is_recorded_and_does_not_end_the_episode(self):
        world = World(task=TASKS[0])
        world.step("open", {"doc": "no-such-doc"})
        self.assertEqual(world.errors, 1)
        self.assertFalse(solved(world))
        world.step("submit", {"answer": TASKS[0].answer})
        self.assertTrue(solved(world))

    def test_the_demonstrations_cover_six_tasks_and_leave_three(self):
        """The headroom is the point: a loop trained only on demonstrated tasks has no gradient."""
        demonstrated = {json.loads(line)["task_id"] for line in DEMOS.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(demonstrated, {task.id for task in DEMO_TASKS})
        self.assertEqual(len(EXTRA_TASKS), 3)
        self.assertEqual(len(TASKS), 9)
        self.assertFalse(demonstrated & {task.id for task in EXTRA_TASKS})

    def test_the_oracle_uses_at_most_the_turn_budget(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                self.assertLessEqual(len(oracle_episode(task).calls), task.max_turns)


class ProtocolTests(unittest.TestCase):
    def test_the_demonstrations_parse(self):
        for line in DEMOS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            call = protocol.parse(row["reply"])
            self.assertIn(call.tool, protocol.TOOLS)

    def test_each_refusal_has_its_own_name(self):
        cases = {
            "": "empty reply",
            "I will look at the report.": "no object",
            '{"tool": "open", "args": {"doc": "a"}} {"tool": "open", "args": {"doc": "b"}}': "not one object",
            'Sure! {"tool": "submit", "args": {"answer": "1"}}': "text around the object",
            '{"tool": "delete", "args": {}}': "unknown tool 'delete'",
            '{"tool": "submit"}': "args is not an object",
            '{"tool": "submit", "args": {"answer": "1"': "bad json",
        }
        for reply, reason in cases.items():
            with self.subTest(reply=reply[:30]):
                with self.assertRaises(protocol.ProtocolError) as caught:
                    protocol.parse(reply)
                if reason == "bad json":
                    # the json module's own wording is not part of the contract
                    self.assertTrue(caught.exception.reason.startswith("bad json:"), caught.exception.reason)
                else:
                    self.assertEqual(caught.exception.reason, reason)

    def test_the_demonstrations_are_current(self):
        done = subprocess.run(
            [sys.executable, str(HERE.parent / "env" / "build_demos.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class AccountingTests(unittest.TestCase):
    def test_a_fresh_batch_is_all_kept(self):
        batch = [{"version": "p5-v4"} for _ in range(4)]
        keep, drop = credit.partition(batch, "p5-v4", bound=0)
        self.assertEqual((len(keep), len(drop)), (4, 0))

    def test_the_bound_is_inclusive(self):
        batch = [{"version": "p5-v3"}, {"version": "p5-v2"}]
        keep, drop = credit.partition(batch, "p5-v4", bound=1)
        self.assertEqual([row["staleness"] for row in keep], [1])
        self.assertEqual([row["staleness"] for row in drop], [2])

    def test_staleness_cannot_go_negative(self):
        keep, _ = credit.partition([{"version": "p5-v9"}], "p5-v4", bound=0)
        self.assertEqual(keep[0]["staleness"], 0)

    def test_summary_is_computed_from_the_log(self):
        history = [{"trained_on": 6, "used": 3, "dropped": 1, "mean_staleness": 0.6, "max_staleness": 1}]
        summary = credit.summarise(history, seconds=120.0)
        self.assertEqual(summary["samples"], 7)
        self.assertEqual(summary["samples_per_minute"], 3.5)
        self.assertEqual(summary["mean_staleness"], 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
