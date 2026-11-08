"""Structural claims about the P6 task set, the gate and the retrieval record.

    python3 p6-memento-skills/checks/check_p6.py

The retrieval numbers in Chapter 12 are quoted from `fixtures/retrieval.json`; this check
recomputes every one of them from the same 60 queries, so a number cannot drift away from the
measurement that produced it. It also asserts the properties the chapter reasons about: the
families are distinct, every family is oracle-solvable, and the library admits a true procedure
and rejects a plausible false one.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from p6_memento_skills.families import FAMILIES, make_instances, score  # noqa: E402
from p6_memento_skills.library import Candidate, Library, validate  # noqa: E402
from p6_memento_skills.retrieval import ROUTERS, SYMPTOMS, SkillDoc, filler_docs, recall_at_1  # noqa: E402

RECORD = ROOT / "fixtures" / "retrieval.json"
SCALES = {"ten": (0, 0), "fifty": (10, 30), "hundred": (10, 80)}
QUERY_SEEDS = range(200, 206)


def pool() -> list[tuple[str, str, str]]:
    instances = make_instances(per_family=max(QUERY_SEEDS) + 1)
    return [
        (family.id, instance.ask, SYMPTOMS[family.id])
        for family in FAMILIES
        for instance in instances
        if instance.family == family.id and instance.seed in QUERY_SEEDS
    ]


def documents(decoys: int, fillers: int) -> list[SkillDoc]:
    real = [SkillDoc(name=f.id, family=f.id, trigger=f.title, procedure=f.skill) for f in FAMILIES]
    wrong = [
        SkillDoc(name=f"decoy-{f.id}", family=f"decoy:{f.id}", trigger=f.title, procedure=f.decoy)
        for f in FAMILIES[:decoys]
    ]
    return real + wrong + filler_docs(fillers, seed=11)


class TaskSetTests(unittest.TestCase):
    def test_ten_families_and_twenty_instances_each(self):
        instances = make_instances(per_family=20)
        self.assertEqual(len(FAMILIES), 10)
        self.assertEqual(len(instances), 200)
        self.assertEqual(len({instance.family for instance in instances}), 10)

    def test_every_instance_has_an_answer(self):
        self.assertTrue(all(instance.answer.strip() for instance in make_instances(per_family=20)))

    def test_the_oracle_agrees_with_itself(self):
        for instance in make_instances(per_family=2):
            self.assertTrue(score(instance, instance.answer))

    def test_families_are_distinguishable(self):
        titles = [family.title for family in FAMILIES]
        self.assertEqual(len(titles), len(set(titles)))
        self.assertEqual(len({family.skill for family in FAMILIES}), 10)


class GateTests(unittest.TestCase):
    def test_a_true_procedure_is_admitted_and_a_decoy_is_not(self):
        holdout = [i for i in make_instances(per_family=6)]

        def policy_for(family_id: str):
            answers = {i.ask: i for i in holdout}

            def policy(ask: str, procedure: str) -> str:
                instance = answers[ask]
                return instance.answer if procedure == next(f.skill for f in FAMILIES if f.id == family_id) else ""

            return policy

        admitted = rejected = 0
        for family in FAMILIES:
            pool_ = [i for i in holdout if i.family == family.id]
            good = Candidate(name="good", family=family.id, trigger=family.title, procedure=family.skill)
            bad = Candidate(name="bad", family=family.id, trigger=family.title, procedure=family.decoy)
            self.assertTrue(validate(good, family, policy_for(family.id), pool_).accepted)
            self.assertFalse(validate(bad, family, policy_for(family.id), pool_).accepted)
            admitted += 1
            rejected += 1
        self.assertEqual((admitted, rejected), (10, 10))

    def test_a_library_keeps_one_skill_per_family(self):
        family = FAMILIES[0]
        holdout = [i for i in make_instances(per_family=6) if i.family == family.id]

        def policy(ask: str, procedure: str) -> str:
            answer = {i.ask: i for i in holdout}[ask].answer
            return answer if procedure == family.skill else ""

        library = Library()
        for name in ("a", "b"):
            library.admit(validate(Candidate(name=name, family=family.id, trigger=family.title, procedure=family.skill), family, policy, holdout))
        self.assertEqual(len(library), 1)


class RecordTests(unittest.TestCase):
    def test_the_record_matches_a_recomputation(self):
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        queries = pool()
        self.assertEqual(record["queries"], len(queries))
        for scale, (decoys, fillers) in SCALES.items():
            docs = documents(decoys, fillers)
            for name, router in ROUTERS.items():
                expected = round(recall_at_1(router, queries, docs), 3)
                self.assertAlmostEqual(record["scales"][scale][name]["recall_at_1"], expected, places=3,
                                       msg=f"{scale}/{name}")
                self.assertEqual(record["scales"][scale][name]["documents"], len(docs))

    def test_the_record_states_what_the_query_contains(self):
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        self.assertIn("failure symptom", record["query_includes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
