"""Tests for common/contamination.py.

Chapter 7 says the contamination check is a script you can re-run, so the script's own
behaviour has to be pinned down - including the two ways it can lie: saying "clean" about
something it could not compare, and letting a similarity score decide a verdict that only
word runs are allowed to decide.
"""

from __future__ import annotations

import unittest

from common.contamination import (
    ContaminationReport,
    char_grams,
    compare,
    jaccard,
    normalise,
    word_runs,
)

LONG = (
    "Janet's ducks lay sixteen eggs per day. She eats three for breakfast every morning and "
    "bakes muffins with four. She sells the rest at the market for two dollars each."
)
LONG_REWORDED_TAIL = LONG.replace("two dollars each", "three dollars each")
#: Same token count as the original head (janet s ducks -> marta s geese), so the runs that do
#: not cover the change still line up.  A substitution of a different length would shift every
#: run and make the test look like a pass for the wrong reason.
LONG_REWORDED_HEAD = LONG.replace("Janet's ducks", "Marta's geese")


def tasks(*items: tuple[str, str]) -> list[tuple[str, str]]:
    return list(items)


class NormaliseTests(unittest.TestCase):
    def test_case_and_punctuation_do_not_matter(self) -> None:
        self.assertEqual(
            normalise("What IS 17 * 23?"),
            normalise("what is 17 23"),
        )

    def test_latex_and_markdown_are_not_part_of_the_problem(self) -> None:
        self.assertEqual(
            normalise("$17 \\times 23$"),
            normalise("17 * 23"),
        )
        self.assertEqual(normalise("**17 & 23**"), "17 23")

    def test_a_laTeX_copy_of_a_task_is_still_a_copy(self) -> None:
        """Markup is the author's; the problem underneath is what gets compared."""
        plain = (
            "The stall sells 16 eggs every morning and 4 muffins in the afternoon, and in the "
            "evening the owner counts the money and writes it down in a small blue notebook."
        )
        latex = plain.replace("16", "$16$").replace("4", "$4$").replace("blue", "**blue**")
        self.assertEqual(normalise(latex), normalise(plain))
        report = compare(tasks(("t01", latex)), [("eval-0", plain)])
        self.assertEqual(report.comparisons[0].verdict, "contaminated")

    def test_number_words_are_not_folded_into_digits(self) -> None:
        """A pinned limit, not a wish.  The similarity column is what shows this case."""
        self.assertNotEqual(normalise("16 eggs"), normalise("sixteen eggs"))

    def test_numbers_survive_normalisation(self) -> None:
        """A shared number is part of a shared problem; dropping digits would hide a copy."""
        self.assertIn("23", normalise("What is 17 * 23?"))


class RunTests(unittest.TestCase):
    def test_a_short_text_has_no_runs(self) -> None:
        self.assertEqual(word_runs("what is 17 23", 13), set())

    def test_a_long_enough_text_has_one_run_per_start(self) -> None:
        words = " ".join(f"w{index}" for index in range(20))
        self.assertEqual(len(word_runs(words, 13)), 20 - 13 + 1)

    def test_char_grams_of_an_empty_text_are_empty(self) -> None:
        self.assertEqual(char_grams("   "), set())
        self.assertEqual(jaccard(set(), {"abc"}), 0.0)


class VerdictTests(unittest.TestCase):
    def test_an_identical_item_is_contaminated(self) -> None:
        report = compare(tasks(("t01", LONG)), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "contaminated")
        self.assertEqual(comparison.containment, 1.0)
        self.assertFalse(report.ok)

    def test_a_copy_reworded_before_the_last_span_still_shares_runs(self) -> None:
        """The realistic leak: a task family written from a benchmark item, not verbatim."""
        report = compare(tasks(("t01", LONG_REWORDED_HEAD)), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertGreater(comparison.shared_runs, 0)
        self.assertEqual(comparison.verdict, "contaminated")

    def test_a_short_reworded_item_escapes_the_word_measure(self) -> None:
        """The tool's own limit, pinned so nobody has to discover it in production.

        A fourteen-word item has two 13-word runs, and both cover a change made in the middle,
        so a lightly reworded short item scores "clean" even though a human would call it a
        copy.  Long documents leak plenty of surviving runs; short items are where the word
        measure goes blind.  This is why the report is read next to the similarity column, and
        why the book does not claim the check proves provenance.
        """
        original = "A robe takes two bolts of blue fiber and half that much white fiber"
        reworded = "A robe takes two bolts of red fiber and half that much white fiber"
        report = compare(tasks(("t01", reworded)), [("eval-0", original)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.runs, 2)
        self.assertEqual(comparison.shared_runs, 0)
        self.assertEqual(comparison.verdict, "clean")
        self.assertGreater(comparison.similarity, 0.5, "the similarity column is the only clue here")

    def test_unrelated_items_are_clean(self) -> None:
        report = compare(
            tasks(("t01", LONG)), [("eval-0", "A robe takes two bolts of blue fiber and half that much white fiber.")]
        )
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "clean")
        self.assertEqual(comparison.shared_runs, 0)
        self.assertTrue(report.ok)

    def test_a_short_task_is_undecided_not_clean(self) -> None:
        report = compare(tasks(("t01", "What is 17 * 23?")), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "undecided")
        self.assertIn("only 4 word(s)", comparison.note)

    def test_similarity_never_decides_a_verdict(self) -> None:
        """Same words, different order: high character overlap, no shared run, no verdict."""
        first = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
        second = "omicron xi nu mu lambda kappa iota theta eta zeta epsilon delta gamma beta alpha"
        report = compare(tasks(("t01", first)), [("eval-0", second)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.shared_runs, 0)
        self.assertEqual(comparison.verdict, "clean")
        self.assertGreater(comparison.similarity, 0.2)

    def test_containment_counts_the_tasks_own_runs(self) -> None:
        """A task that is a copy plus an instruction has most, not all, of its runs shared."""
        padded = "Solve this problem. " + LONG + " Show your working and state the answer."
        report = compare(tasks(("t01", padded)), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "contaminated")
        self.assertLess(comparison.containment, 1.0)
        self.assertGreater(comparison.containment, 0.5)

    def test_the_best_match_is_named(self) -> None:
        report = compare(
            tasks(("t01", LONG)),
            [("eval-0", "unrelated"), ("eval-1", LONG), ("eval-2", "also unrelated")],
        )
        (comparison,) = report.comparisons
        self.assertEqual(comparison.best_match, "eval-1")

    def test_a_longer_problem_still_matches_a_copied_sentence_inside_it(self) -> None:
        """Length never filters containment: the copied sentence can sit in a longer item."""
        long_item = LONG + " " + " ".join(f"filler{index}" for index in range(40))
        report = compare(tasks(("t01", LONG)), [("eval-0", long_item)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "contaminated")


class ShallowEvidenceTests(unittest.TestCase):
    """A clean verdict on one run is clean, but it is not strong - and it has to say so."""

    SHORTISH = "The stall sells 16 fresh eggs every single morning and nothing else at all today"

    def test_a_short_but_decidable_item_is_flagged_as_shallow(self) -> None:
        report = compare(tasks(("t01", self.SHORTISH)), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "clean")
        self.assertLess(comparison.runs, 5)
        self.assertIn("shallow", comparison.note)
        self.assertEqual([c.task_id for c in report.shallow], ["t01"])
        self.assertIn("fewer than", report.render())

    def test_a_long_clean_item_is_not_flagged(self) -> None:
        long_clean = " ".join(f"word{index}" for index in range(30))
        report = compare(tasks(("t01", long_clean)), [("eval-0", LONG)])
        (comparison,) = report.comparisons
        self.assertEqual(comparison.verdict, "clean")
        self.assertGreaterEqual(comparison.runs, 5)
        self.assertEqual(comparison.note, "")
        self.assertEqual(report.shallow, [])

    def test_a_clean_verdict_names_no_match(self) -> None:
        """Zero shared runs is not a match: naming the tie-break would invent a finding."""
        report = compare(tasks(("t01", self.SHORTISH)), [("eval-0", LONG), ("eval-1", LONG)])
        (comparison,) = report.comparisons
        self.assertIsNone(comparison.best_match)

    def test_an_undecided_item_is_not_counted_as_shallow(self) -> None:
        report = compare(tasks(("t01", "What is 17 * 23?")), [("eval-0", LONG)])
        self.assertEqual(report.undecided[0].verdict, "undecided")
        self.assertEqual(report.shallow, [])

    def test_the_shallow_count_reaches_the_json_report(self) -> None:
        report = compare(tasks(("t01", self.SHORTISH)), [("eval-0", LONG)])
        self.assertEqual(report.to_dict()["shallow"], ["t01"])


class ReportTests(unittest.TestCase):
    def test_the_digest_of_the_evaluation_file_travels_with_the_report(self) -> None:
        report = compare(
            tasks(("t01", LONG)),
            [("eval-0", LONG)],
            eval_source="datasets/payloads/gsm8k/rows/test.jsonl",
            eval_digest="sha256:deadbeef",
        )
        payload = report.to_dict()
        self.assertEqual(payload["eval_digest"], "sha256:deadbeef")
        self.assertEqual(payload["eval_source"], "datasets/payloads/gsm8k/rows/test.jsonl")
        self.assertEqual(payload["contaminated"], ["t01"])

    def test_an_empty_evaluation_set_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            compare(tasks(("t01", LONG)), [])
        self.assertIn("would report clean", str(caught.exception))

    def test_the_report_does_not_call_an_uncomparable_set_clean(self) -> None:
        report = compare(tasks(("t01", "What is 17 * 23?")), [("eval-0", LONG)])
        self.assertNotIn("nothing overlaps", report.render())
        self.assertIn("nothing comparable", report.render())

    def test_a_mixed_report_names_both_counts(self) -> None:
        report = compare(
            tasks(("t01", LONG), ("t02", "What is 17 * 23?")),
            [("eval-0", LONG)],
            eval_source="eval.jsonl",
            eval_digest="sha256:x",
        )
        rendered = report.render()
        self.assertIn("contaminated: 1", rendered)
        self.assertIn("undecided: 1", rendered)
        self.assertIn("1 undecided", rendered)

    def test_an_empty_task_set_is_not_a_failure(self) -> None:
        report: ContaminationReport = compare([], [("eval-0", LONG)])
        self.assertTrue(report.ok)
        self.assertEqual(report.comparisons, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
