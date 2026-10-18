"""The token-level objective: sign, masking, and length independence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for path in (str(REPO), str(HERE.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p2_reagent.rl_loss import TurnObjective, distinct_weighted_turns, objective  # noqa: E402


class ObjectiveTests(unittest.TestCase):
    def test_a_zero_weight_contributes_nothing(self) -> None:
        turn = TurnObjective(0.0, (-5.0, -5.0, -5.0))
        self.assertFalse(turn.counted)
        self.assertEqual(turn.contribution, 0.0)
        self.assertEqual(objective([turn]), (0.0, 0))

    def test_the_sign_is_the_one_that_trains(self) -> None:
        worse = TurnObjective(1.0, (-3.0,))
        better = TurnObjective(1.0, (-0.5,))
        self.assertGreater(worse.contribution, better.contribution)
        self.assertLess(objective([better])[0], objective([worse])[0])

    def test_length_does_not_outweigh_quality(self) -> None:
        short = TurnObjective(1.0, (-1.0,))
        long = TurnObjective(1.0, tuple([-1.0] * 40))
        self.assertAlmostEqual(short.contribution, long.contribution)

    def test_the_mean_is_over_weighted_turns_only(self) -> None:
        turns = [TurnObjective(1.0, (-2.0,)), TurnObjective(0.0, (-9.0,)), TurnObjective(-1.0, (-2.0,))]
        value, counted = objective(turns)
        self.assertEqual(counted, 2)
        self.assertAlmostEqual(value, 0.0)
        self.assertEqual(distinct_weighted_turns(turns), 2)

    def test_an_empty_round_is_zero_not_an_error(self) -> None:
        self.assertEqual(objective([]), (0.0, 0))

    def test_a_turn_with_no_tokens_is_not_counted(self) -> None:
        self.assertEqual(objective([TurnObjective(0.7, ())]), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
