"""The prompt shape and the parser: the two places format compliance is won or lost.

These are separate from the loop's tests because they fail for different reasons. A loop test
fails when the machinery is wrong; these fail when the interface to the model is wrong, and the
interface is what the fine-tune is judged on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p2_reagent import actions, env, prompt  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_the_menu_is_shown_verbatim_and_the_slot_is_named(self) -> None:
        attempt = env.start("files-delete-count")
        messages = prompt.render(attempt.task, attempt.observation, attempt.menu())
        self.assertEqual(messages[0]["role"], "system")
        text = messages[-1]["content"]
        for action in attempt.menu():
            self.assertIn(f"- {action}", text)
        self.assertIn("ACTION: <action>", text)

    def test_the_prompt_does_not_leak_the_answer(self) -> None:
        for task in env.all_tasks():
            attempt = env.start(task.task_id)
            text = prompt.render(attempt.task, attempt.observation, attempt.menu())[-1]["content"]
            self.assertNotIn("Answer:", text)
            self.assertNotIn("expected", text.lower())

    def test_history_is_rendered_when_given(self) -> None:
        attempt = env.start("ledger-net-seven")
        messages = prompt.render(
            attempt.task, attempt.observation, attempt.menu(), [("add 12", "ok", "balance: 12")]
        )
        self.assertIn("- add 12 -> ok: balance: 12", messages[-1]["content"])


class ParserTests(unittest.TestCase):
    MENU = ("ls", "rm tmp.log", "done 2")

    def test_a_plain_line_is_read(self) -> None:
        parsed = actions.parse_action("ACTION: rm tmp.log", self.MENU)
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.action, "rm tmp.log")

    def test_decoration_case_and_spacing_are_tolerated(self) -> None:
        for text in (
            "**ACTION:** `rm tmp.log`",
            "action:   rm   tmp.log",
            "ACTION: rm tmp.log.",
            '"ACTION: rm tmp.log"',
        ):
            with self.subTest(text=text):
                parsed = actions.parse_action(text, self.MENU)
                self.assertTrue(parsed.ok, parsed)
                self.assertEqual(parsed.action, "rm tmp.log")

    def test_the_last_line_is_not_needed_but_a_second_answer_is_refused(self) -> None:
        parsed = actions.parse_action("ACTION: ls\nACTION: rm tmp.log", self.MENU)
        self.assertEqual(parsed.reason, "ambiguous")
        self.assertIsNone(parsed.action)

    def test_prose_is_not_guessed_at(self) -> None:
        parsed = actions.parse_action("I would remove tmp.log first.", self.MENU)
        self.assertEqual(parsed.reason, "no_action_line")
        self.assertIsNone(parsed.action)
        self.assertEqual(parsed.raw, "I would remove tmp.log first.")

    def test_an_invented_action_is_reported_with_what_was_said(self) -> None:
        parsed = actions.parse_action("ACTION: sudo rm -rf /", self.MENU)
        self.assertEqual(parsed.reason, "not_in_menu")
        self.assertEqual(parsed.raw, "sudo rm -rf /")

    def test_an_empty_reply_has_its_own_reason(self) -> None:
        self.assertEqual(actions.parse_action("", self.MENU).reason, "no_action_line")
        self.assertEqual(actions.parse_action("ACTION:", self.MENU).reason, "empty")

    def test_a_menu_entry_copy_is_accepted_even_with_odd_spacing(self) -> None:
        parsed = actions.parse_action("ACTION: done   2", ("done 2",))
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.action, "done 2")


class TaskSetTests(unittest.TestCase):
    def test_every_specification_is_internally_consistent(self) -> None:
        for task in env.all_tasks():
            with self.subTest(task=task.task_id):
                self.assertIsNotNone(task.spec)
                self.assertIsNone(task.spec.contradiction())
                self.assertEqual(task.spec.expected, task.spec.recomputed_answer())

    def test_the_set_is_two_families_of_at_least_ten(self) -> None:
        counts = {}
        for task in env.all_tasks():
            counts[task.family] = counts.get(task.family, 0) + 1
        self.assertEqual(sorted(counts), ["files", "ledger"])
        for family, count in counts.items():
            self.assertGreaterEqual(count, 10, family)

    def test_the_menu_always_contains_the_correct_action(self) -> None:
        for task in env.all_tasks():
            self.assertIn(task.spec.correct_action if task.family == "files" else task.spec.menu()[0],
                          task.spec.menu())


if __name__ == "__main__":
    unittest.main()
