"""Fusion: weights, gates, vetoes, and the conflicts worth reading."""

from __future__ import annotations

import unittest

from common.fusion import FusionPolicy, detect_conflicts, fuse, fuse_with_conflicts


class PolicyValidationTests(unittest.TestCase):
    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FusionPolicy(weights={"rule": 1.0}, mode="magic")

    def test_gated_mode_needs_both_terms(self) -> None:
        with self.assertRaises(ValueError):
            FusionPolicy(weights={"rule": 1.0}, mode="gated", gate="rule")

    def test_veto_mode_needs_a_veto_term(self) -> None:
        with self.assertRaises(ValueError):
            FusionPolicy(weights={"rule": 1.0}, mode="veto")

    def test_a_policy_needs_at_least_one_weight(self) -> None:
        with self.assertRaises(ValueError):
            FusionPolicy(weights={})


class WeightedFusionTests(unittest.TestCase):
    def test_total_and_contributions(self) -> None:
        policy = FusionPolicy(weights={"rule": 1.0, "judge": 0.5})
        result = fuse({"rule": 1.0, "judge": 0.2}, policy)
        self.assertAlmostEqual(result.total, 1.1)
        self.assertAlmostEqual(result.contributions["judge"], 0.1)
        self.assertIsNone(result.vetoed_by)

    def test_an_unweighted_term_is_an_error_not_a_silent_zero(self) -> None:
        policy = FusionPolicy(weights={"rule": 1.0})
        with self.assertRaises(KeyError) as ctx:
            fuse({"rule": 1.0, "judge": 0.3}, policy)
        self.assertIn("judge", str(ctx.exception))

    def test_a_weighted_term_that_was_not_supplied_is_an_error(self) -> None:
        policy = FusionPolicy(weights={"rule": 1.0, "judge": 0.5})
        with self.assertRaises(ValueError):
            fuse({"rule": 1.0}, policy)

    def test_render_lists_every_contribution(self) -> None:
        result = fuse({"rule": 1.0, "judge": -0.2}, FusionPolicy(weights={"rule": 1.0, "judge": 1.0}))
        text = result.render()
        self.assertIn("total=", text)
        self.assertIn("judge=", text)


class GatedFusionTests(unittest.TestCase):
    policy = FusionPolicy(weights={"format": 1.0, "judge": 1.0}, mode="gated",
                          gate="format", gated="judge", gate_threshold=1.0)

    def test_closed_gate_zeroes_the_gated_term(self) -> None:
        result = fuse({"format": 0.0, "judge": 1.0}, self.policy)
        self.assertAlmostEqual(result.total, 0.0)
        self.assertAlmostEqual(result.contributions["judge"], 0.0)

    def test_open_gate_lets_it_through(self) -> None:
        result = fuse({"format": 1.0, "judge": 1.0}, self.policy)
        self.assertAlmostEqual(result.total, 2.0)

    def test_gating_on_a_missing_term_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            fuse({"judge": 1.0}, self.policy)


class VetoFusionTests(unittest.TestCase):
    policy = FusionPolicy(weights={"safety": 1.0, "judge": 1.0}, mode="veto", veto_on=("safety",))

    def test_a_negative_veto_term_zeroes_the_total_and_says_why(self) -> None:
        result = fuse({"safety": -1.0, "judge": 1.0}, self.policy)
        self.assertAlmostEqual(result.total, 0.0)
        self.assertEqual(result.vetoed_by, "safety")
        self.assertIn("vetoed by safety", result.render())

    def test_a_non_negative_veto_term_does_not_fire(self) -> None:
        result = fuse({"safety": 0.0, "judge": 1.0}, self.policy)
        self.assertAlmostEqual(result.total, 1.0)
        self.assertIsNone(result.vetoed_by)

    def test_vetoing_on_a_missing_term_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            fuse({"judge": 1.0}, self.policy)


class ConflictTests(unittest.TestCase):
    sources = {"rule": ["exact"], "judge": ["judge"]}

    def test_opposite_signs_above_the_margin_conflict(self) -> None:
        conflicts = detect_conflicts({"exact": 1.0, "judge": -0.3}, self.sources)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("disagrees", conflicts[0])

    def test_a_small_disagreement_is_not_a_conflict(self) -> None:
        self.assertEqual(detect_conflicts({"exact": 1.0, "judge": -0.1}, self.sources), [])

    def test_agreement_is_not_a_conflict(self) -> None:
        self.assertEqual(detect_conflicts({"exact": 1.0, "judge": 0.4}, self.sources), [])

    def test_terms_from_the_same_source_never_conflict(self) -> None:
        sources = {"rule": ["exact", "format"]}
        self.assertEqual(detect_conflicts({"exact": 1.0, "format": -0.5}, sources), [])

    def test_missing_terms_are_skipped_not_invented(self) -> None:
        self.assertEqual(detect_conflicts({"exact": 1.0}, self.sources), [])

    def test_negative_margin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect_conflicts({"exact": 1.0, "judge": -0.3}, self.sources, margin=-1.0)


class CombinedFusionTests(unittest.TestCase):
    def test_conflicts_ride_along_with_the_total(self) -> None:
        policy = FusionPolicy(weights={"exact": 1.0, "judge": 1.0})
        result = fuse_with_conflicts(
            {"exact": 1.0, "judge": -0.4}, policy, {"rule": ["exact"], "judge": ["judge"]}
        )
        self.assertAlmostEqual(result.total, 0.6)
        self.assertEqual(len(result.conflicts), 1)


if __name__ == "__main__":
    unittest.main()
