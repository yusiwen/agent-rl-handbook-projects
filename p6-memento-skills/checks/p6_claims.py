"""The numbers Chapter 12 quotes, recomputed from `fixtures/outcome.json`.

The claim this file exists to protect is not a single success rate: it is the *shape* of the result -
some families improved, two that already worked collapsed, and the total went down. A proofread
cannot catch a table that quietly reports only the improving column; this can.

    python3 p6-memento-skills/checks/p6_claims.py
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent / "fixtures" / "outcome.json"
CONDITIONS = ("bare", "oracle-skill", "keyword", "idf")


def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def wilson(solved: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = solved / total
    z = 1.96
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (round((centre - half) * 100, 1), round((centre + half) * 100, 1))


class OutcomeTests(unittest.TestCase):
    def test_every_condition_ran_on_the_same_instances(self):
        data = record()
        counts = {condition: data["conditions"][condition]["instances"] for condition in CONDITIONS}
        self.assertEqual(set(counts.values()), {100})
        self.assertEqual(len(data["rows"]), 400)

    def test_the_four_totals_are_what_the_records_say(self):
        data = record()
        totals = {condition: data["conditions"][condition]["solved"] for condition in CONDITIONS}
        self.assertEqual(totals, {"bare": 34, "oracle-skill": 25, "keyword": 27, "idf": 29})

    def test_no_injection_beat_the_question_alone(self):
        """The chapter's headline: the ceiling was below the floor."""
        data = record()
        bare = data["conditions"]["bare"]["solved"]
        for condition in ("oracle-skill", "keyword", "idf"):
            self.assertLess(data["conditions"][condition]["solved"], bare, condition)

    def test_the_intervals_overlap_so_the_total_is_a_direction_not_an_effect(self):
        data = record()
        low, high = wilson(data["conditions"]["bare"]["solved"], 100)
        overlaps = 0
        for condition in ("oracle-skill", "keyword", "idf"):
            other = wilson(data["conditions"][condition]["solved"], 100)
            if not (other[1] < low or other[0] > high):
                overlaps += 1
        self.assertEqual(overlaps, 3)

    def test_two_families_that_worked_broke(self):
        data = record()
        per = data["conditions"]
        self.assertEqual(per["bare"]["per_family"]["split-remainder"], 10)
        self.assertLessEqual(per["idf"]["per_family"]["split-remainder"], 4)
        self.assertEqual(per["bare"]["per_family"]["dedupe-keep-last"], 10)
        self.assertLessEqual(per["idf"]["per_family"]["dedupe-keep-last"], 7)

    def test_one_family_that_failed_improved(self):
        data = record()
        per = data["conditions"]
        self.assertEqual(per["bare"]["per_family"]["pivot-counts"], 0)
        for condition in ("oracle-skill", "keyword", "idf"):
            self.assertGreaterEqual(per[condition]["per_family"]["pivot-counts"], 4, condition)

    def test_retrieval_was_not_the_bottleneck(self):
        data = record()
        self.assertEqual(data["routing"]["idf"], 1.0)
        idf_rows = [row for row in data["rows"] if row["condition"] == "idf"]
        self.assertEqual(sum(1 for row in idf_rows if row["retrieved"] == row["family"]), len(idf_rows))

    def test_the_truncation_is_reported_not_hidden(self):
        data = record()
        errors = {condition: data["conditions"][condition]["errors"] for condition in CONDITIONS}
        self.assertTrue(all(count > 20 for count in errors.values()), errors)
        self.assertEqual(errors["bare"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
