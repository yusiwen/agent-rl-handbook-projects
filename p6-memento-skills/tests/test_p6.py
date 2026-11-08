import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from p6_memento_skills.families import FAMILIES, FAMILY_BY_ID, make_instances, score
from p6_memento_skills.library import Candidate, Library, validate
from p6_memento_skills.retrieval import ROUTERS, SYMPTOMS, SkillDoc, recall_at_1


def scripted_policy(family_id: str):
    """A stand-in for a model: it answers correctly only when given the true procedure.

    The gate is supposed to admit a text because it *works*, not because it reads well, so the
    test supplies a policy whose behaviour depends on the text and nothing else. A real run
    replaces this with the served model; the gate's logic is identical either way.
    """

    def policy(ask: str, procedure: str) -> str:
        instance = ANSWERS.get(ask)
        if instance is None or instance.family != family_id:
            raise KeyError(f"the policy was asked something it was not given: {ask[:40]}")
        return instance.answer if procedure == FAMILY_BY_ID[family_id].skill else ""

    return policy


INSTANCES = make_instances(per_family=6)
ANSWERS = {instance.ask: instance for instance in INSTANCES}


def docs() -> list[SkillDoc]:
    return [SkillDoc(name=family.id, family=family.id, trigger=family.title, procedure=family.skill) for family in FAMILIES]


class FamilyTests(unittest.TestCase):
    def test_every_family_has_an_oracle_that_answers(self):
        instances = make_instances(per_family=3)
        self.assertEqual(len(instances), 30)
        self.assertTrue(all(instance.answer.strip() for instance in instances))

    def test_instances_are_reproducible(self):
        first = make_instances(per_family=2)
        second = make_instances(per_family=2)
        self.assertEqual([instance.ask for instance in first], [instance.ask for instance in second])
        self.assertEqual([instance.answer for instance in first], [instance.answer for instance in second])

    def test_answers_within_a_family_are_not_all_identical(self):
        """A family whose six instances share one answer cannot teach anything by retrieval."""
        for family in FAMILIES:
            answers = {instance.answer for instance in make_instances(per_family=6) if instance.family == family.id}
            self.assertGreaterEqual(len(answers), 3, family.id)

    def test_asks_do_not_repeat_across_families(self):
        asks = [instance.ask for instance in make_instances(per_family=2)]
        self.assertEqual(len(asks), len(set(asks)))

    def test_score_ignores_surrounding_whitespace(self):
        instance = make_instances(per_family=1)[0]
        self.assertTrue(score(instance, f"  {instance.answer}\n"))
        self.assertFalse(score(instance, instance.answer + " x"))

    def test_every_family_carries_a_true_and_a_false_procedure(self):
        for family in FAMILIES:
            self.assertTrue(family.skill.strip())
            self.assertTrue(family.decoy.strip())
            self.assertNotEqual(family.skill, family.decoy)
            self.assertIn(family.id, SYMPTOMS)


class RetrievalTests(unittest.TestCase):
    def test_every_router_ranks_every_document(self):
        pool = [(family.id, instance.ask, SYMPTOMS[family.id]) for family in FAMILIES for instance in INSTANCES if instance.family == family.id][:6]
        for router in ROUTERS.values():
            ranked = router.rank(pool[0][1], pool[0][2], docs())
            self.assertEqual(len(ranked), len(FAMILIES))
            self.assertGreaterEqual(ranked[0][1], ranked[-1][1])

    def test_the_true_skill_is_findable_without_decoys(self):
        pool = [(family.id, instance.ask, SYMPTOMS[family.id]) for family in FAMILIES for instance in INSTANCES if instance.family == family.id]
        self.assertEqual(recall_at_1(ROUTERS["idf"], pool, docs()), 1.0)

    def test_a_symptom_is_recorded_for_every_family(self):
        self.assertEqual(len(SYMPTOMS), len(FAMILIES))


class GateTests(unittest.TestCase):
    def test_the_true_procedure_is_admitted(self):
        for family in FAMILIES:
            candidate = Candidate(name=f"{family.id}-v1", family=family.id, trigger=family.title, procedure=family.skill)
            holdout = [instance for instance in make_instances(per_family=6) if instance.family == family.id]
            verdict = validate(candidate, family, scripted_policy(family.id), holdout)
            self.assertTrue(verdict.accepted, (family.id, verdict.reason))
            self.assertEqual(verdict.baseline, 0)
            self.assertEqual(verdict.with_skill, len(holdout))

    def test_a_plausible_wrong_procedure_is_rejected(self):
        for family in FAMILIES:
            candidate = Candidate(name=f"{family.id}-decoy", family=family.id, trigger=family.title, procedure=family.decoy)
            holdout = [instance for instance in make_instances(per_family=6) if instance.family == family.id]
            verdict = validate(candidate, family, scripted_policy(family.id), holdout)
            self.assertFalse(verdict.accepted, family.id)
            self.assertIn("no gain", verdict.reason)

    def test_a_family_without_holdout_is_rejected_rather_than_crashing(self):
        family = FAMILIES[0]
        candidate = Candidate(name="x", family=family.id, trigger=family.title, procedure=family.skill)
        verdict = validate(candidate, family, scripted_policy(family.id), [])
        self.assertFalse(verdict.accepted)
        self.assertIn("no held-out", verdict.reason)

    def test_the_incumbent_survives_a_tie(self):
        family = FAMILIES[0]
        holdout = [instance for instance in make_instances(per_family=6) if instance.family == family.id]
        library = Library()
        first = Candidate(name="first", family=family.id, trigger=family.title, procedure=family.skill)
        self.assertTrue(library.admit(validate(first, family, scripted_policy(family.id), holdout)))
        second = Candidate(name="second", family=family.id, trigger=family.title, procedure=family.skill)
        self.assertFalse(library.admit(validate(second, family, scripted_policy(family.id), holdout)))
        self.assertEqual(len(library), 1)
        self.assertEqual(len(library.rejected), 1)

    def test_the_injection_is_empty_until_a_skill_is_admitted(self):
        family = FAMILIES[0]
        holdout = [instance for instance in make_instances(per_family=6) if instance.family == family.id]
        library = Library()
        self.assertEqual(library.injection(family.id), "")
        candidate = Candidate(name="v1", family=family.id, trigger=family.title, procedure=family.skill)
        library.admit(validate(candidate, family, scripted_policy(family.id), holdout))
        injected = library.injection(family.id)
        self.assertIn("Procedure to follow", injected)
        self.assertIn(family.skill, injected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
