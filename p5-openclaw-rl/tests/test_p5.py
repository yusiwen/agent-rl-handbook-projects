"""Unit tests for P5: the world, the protocol, and the accounting of an asynchronous loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from p5_openclaw import credit, protocol  # noqa: E402
from p5_openclaw.world import TASKS, World, normalise, oracle_episode, solved  # noqa: E402


class WorldTests(unittest.TestCase):
    def test_every_task_is_solvable_by_its_oracle(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                self.assertTrue(solved(oracle_episode(task)))

    def test_a_world_starts_empty(self):
        first = World(task=TASKS[0])
        second = World(task=TASKS[0])
        first.step("open", {"doc": "q3-north"})
        self.assertEqual(second.calls, [])

    def test_search_finds_documents_by_word(self):
        world = World(task=TASKS[0])
        observation = world.step("search", {"query": "north"})
        self.assertIn("q3-north", observation.text)

    def test_calc_refuses_anything_but_arithmetic(self):
        world = World(task=TASKS[0])
        observation = world.step("calc", {"expression": "__import__('os').system('true')"})
        self.assertTrue(observation.error)
        self.assertEqual(world.errors, 1)

    def test_an_unknown_document_is_an_error_not_a_crash(self):
        world = World(task=TASKS[0])
        self.assertTrue(world.step("open", {"doc": "nope"}).error)

    def test_answers_are_compared_as_numbers(self):
        self.assertEqual(normalise("5,400.0"), normalise("5400.00"))
        self.assertEqual(normalise("North."), "north")


class ProtocolTests(unittest.TestCase):
    def test_a_single_object_parses(self):
        call = protocol.parse('{"tool": "open", "args": {"doc": "q3-north"}}')
        self.assertEqual(call.tool, "open")
        self.assertEqual(call.args["doc"], "q3-north")

    def test_a_fenced_object_is_accepted(self):
        self.assertEqual(protocol.parse('```json\n{"tool": "submit", "args": {"answer": "1"}}\n```').tool, "submit")

    def test_prose_is_refused_by_name(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse("I will open the northern report now.")
        self.assertEqual(caught.exception.reason, "no object")

    def test_two_objects_are_refused_rather_than_merged(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse('{"tool": "open", "args": {"doc": "a"}} {"tool": "submit", "args": {"answer": "1"}}')
        self.assertEqual(caught.exception.reason, "not one object")

    def test_an_unknown_tool_is_refused(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse('{"tool": "rm", "args": {}}')
        self.assertIn("unknown tool", caught.exception.reason)

    def test_text_around_the_object_is_refused(self):
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse('Sure: {"tool": "submit", "args": {"answer": "1"}}')
        self.assertEqual(caught.exception.reason, "text around the object")


class CreditTests(unittest.TestCase):
    def test_version_index_survives_a_strange_name(self):
        self.assertEqual(credit.version_index("p5-v12"), 12)
        self.assertEqual(credit.version_index("base"), 0)

    def test_staleness_is_version_distance(self):
        self.assertEqual(credit.staleness(["p5-v1", "p5-v3"], "p5-v3"), [2, 0])

    def test_partition_counts_what_it_drops(self):
        batch = [{"version": "p5-v1"}, {"version": "p5-v3"}, {"version": "p5-v0"}]
        keep, drop = credit.partition(batch, "p5-v3", bound=1)
        self.assertEqual(len(keep), 1)
        self.assertEqual(len(drop), 2)
        self.assertEqual([row["staleness"] for row in batch], [2, 0, 3])

    def test_advantages_are_centred(self):
        self.assertEqual(credit.group_advantages([1.0, 0.0]), [0.5, -0.5])
        self.assertEqual(credit.group_advantages([1.0, 1.0]), [0.0, 0.0])

    def test_summary_counts_dropped_samples(self):
        history = [
            {"trained_on": 4, "used": 2, "dropped": 2, "mean_staleness": 0.5, "max_staleness": 1},
            {"trained_on": 6, "used": 3, "dropped": 0, "mean_staleness": 1.0, "max_staleness": 2},
        ]
        summary = credit.summarise(history, seconds=60.0)
        self.assertEqual(summary["samples"], 12)
        self.assertEqual(summary["dropped"], 2)
        self.assertEqual(summary["max_staleness"], 2)
        self.assertEqual(summary["samples_per_minute"], 12.0)


if __name__ == "__main__":
    unittest.main()
