"""The verifier protocol, the registry, and the coverage report."""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from common.verifier import CoverageReport, MissingVerifier, Outcome, VerifierRegistry


class ExactMatch:
    """A tiny verifier, used here as a stand-in for a project's real one."""

    name = "exact_match"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        return "answer" in task

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:
        ok = response.strip() == str(task.get("answer", "")).strip()
        return Outcome(passed=ok, score=1.0 if ok else 0.0, detail="exact match")


class NoScope:
    """Declares itself able to check nothing - the case coverage has to catch."""

    name = "declines_everything"

    def can_verify(self, task: Mapping[str, Any]) -> bool:
        return False

    def verify(self, task: Mapping[str, Any], response: str) -> Outcome:  # pragma: no cover
        raise AssertionError("should never be called")


class Broken:
    name = "broken"


class OutcomeTests(unittest.TestCase):
    def test_score_outside_the_convention_is_rejected(self) -> None:
        for score in (-1.5, 1.5, 42.0):
            with self.assertRaises(ValueError):
                Outcome(passed=True, score=score)

    def test_bounds_are_inclusive(self) -> None:
        self.assertEqual(Outcome(passed=True, score=-1.0).score, -1.0)
        self.assertEqual(Outcome(passed=True, score=1.0).score, 1.0)

    def test_as_terms_prefixes_named_terms_and_keeps_the_score(self) -> None:
        outcome = Outcome(passed=True, score=1.0, terms={"format": 0.5})
        self.assertEqual(outcome.as_terms("rule"), {"rule/format": 0.5, "rule": 1.0})

    def test_as_terms_without_named_terms_is_just_the_score(self) -> None:
        self.assertEqual(Outcome(passed=False, score=0.0).as_terms("rule"), {"rule": 0.0})


class RegistryTests(unittest.TestCase):
    def test_register_and_list(self) -> None:
        registry = VerifierRegistry([ExactMatch()])
        self.assertEqual(registry.names, ["exact_match"])

    def test_duplicate_names_are_rejected(self) -> None:
        registry = VerifierRegistry([ExactMatch()])
        with self.assertRaises(ValueError):
            registry.register(ExactMatch())

    def test_a_verifier_without_a_name_is_rejected(self) -> None:
        class Nameless:
            name = ""

            def can_verify(self, task: Mapping[str, Any]) -> bool:  # pragma: no cover
                return True

            def verify(self, task: Mapping[str, Any], response: str) -> Outcome:  # pragma: no cover
                return Outcome(True, 0.0)

        with self.assertRaises(ValueError):
            VerifierRegistry().register(Nameless())  # type: ignore[arg-type]

    def test_a_verifier_without_verify_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VerifierRegistry().register(Broken())  # type: ignore[arg-type]

    def test_get_unknown_name_raises_and_lists_what_exists(self) -> None:
        registry = VerifierRegistry([ExactMatch()])
        with self.assertRaises(MissingVerifier) as ctx:
            registry.get("nope")
        self.assertIn("exact_match", str(ctx.exception))

    def test_for_task_uses_the_verifier_field(self) -> None:
        registry = VerifierRegistry([ExactMatch()])
        self.assertIsInstance(registry.for_task({"id": "t1", "verifier": "exact_match"}), ExactMatch)

    def test_for_task_without_a_verifier_field_is_missing_not_silent(self) -> None:
        with self.assertRaises(MissingVerifier):
            VerifierRegistry([ExactMatch()]).for_task({"id": "t1"})

    def test_for_task_with_an_unknown_name_is_missing(self) -> None:
        with self.assertRaises(MissingVerifier):
            VerifierRegistry([ExactMatch()]).for_task({"id": "t1", "verifier": "ghost"})


class CoverageTests(unittest.TestCase):
    def tasks(self) -> list[dict[str, Any]]:
        return [
            {"id": "t1", "verifier": "exact_match", "answer": "42"},
            {"id": "t2", "verifier": "ghost"},
            {"id": "t3", "verifier": "declines_everything", "answer": "42"},
            {"id": "t4"},
        ]

    def test_report_counts_every_kind_of_gap(self) -> None:
        registry = VerifierRegistry([ExactMatch(), NoScope()])
        report = registry.coverage(self.tasks())
        self.assertIsInstance(report, CoverageReport)
        self.assertEqual(report.total, 4)
        self.assertEqual(report.covered, 1)
        self.assertEqual(len(report.no_verifier), 2)  # t2 wants a ghost, t4 names nothing
        self.assertEqual(len(report.declined), 1)     # t3 declared a verifier that said no
        self.assertAlmostEqual(report.fraction, 0.25)
        self.assertEqual(report.by_verifier, {"exact_match": 1})

    def test_render_names_the_gaps(self) -> None:
        registry = VerifierRegistry([ExactMatch(), NoScope()])
        text = registry.coverage(self.tasks()).render()
        self.assertIn("1/4 tasks (25%)", text)
        self.assertIn("t2", text)
        self.assertIn("declined", text)

    def test_a_fully_covered_set_says_so(self) -> None:
        registry = VerifierRegistry([ExactMatch()])
        report = registry.coverage([{"id": "t1", "verifier": "exact_match", "answer": "1"}])
        self.assertIn("nothing uncovered", report.render())

    def test_empty_task_set_is_zero_not_a_crash(self) -> None:
        report = VerifierRegistry([ExactMatch()]).coverage([])
        self.assertEqual((report.total, report.covered), (0, 0))
        self.assertEqual(report.fraction, 0.0)


class RuntimeProtocolTests(unittest.TestCase):
    def test_structural_checking_accepts_a_conforming_verifier(self) -> None:
        from common.verifier import Verifier

        self.assertIsInstance(ExactMatch(), Verifier)

    def test_registry_rejects_a_non_conforming_object(self) -> None:
        with self.assertRaises(ValueError):
            VerifierRegistry().register(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
