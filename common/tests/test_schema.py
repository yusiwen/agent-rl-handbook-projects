"""The rollout row: validation, derived quantities, round-tripping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from common.schema import ROLES, Rollout, Segment, read_rollouts, write_rollouts


def trace() -> list[Segment]:
    return [
        Segment(role="system", text="sys", tokens=5),
        Segment(role="user", text="hi", tokens=10),
        Segment(role="model", text="thinking", tokens=20),
        Segment(role="tool", text="42", tokens=5),
        Segment(role="model", text="the answer is 42", tokens=15),
    ]


def rollout(**overrides: object) -> Rollout:
    defaults: dict[str, object] = {
        "episode_id": "e1",
        "prompt_id": "p1",
        "sample_index": 0,
        "group_id": "p1",
        "segments": trace(),
        "reward_terms": {"rule": 1.0, "judge": -0.25},
        "env_version": "fixtures/v1",
        "config_hash": "sha256:aaa",
        "code_sha": "git:bbb",
        "mask": [0] * 15 + [1] * 20 + [0] * 5 + [1] * 15,
        "turn_count": 2,
    }
    defaults.update(overrides)
    return Rollout(**defaults)  # type: ignore[arg-type]


class SegmentTests(unittest.TestCase):
    def test_roles_are_the_ones_the_book_names(self) -> None:
        self.assertEqual(ROLES, ("system", "user", "model", "tool"))

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Segment(role="assistant", text="x", tokens=1)  # type: ignore[arg-type]
        self.assertIn("assistant", str(ctx.exception))

    def test_negative_token_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Segment(role="model", text="x", tokens=-1)


class RolloutTests(unittest.TestCase):
    def test_token_totals(self) -> None:
        row = rollout()
        self.assertEqual(row.total_tokens, 55)
        self.assertEqual(row.model_tokens, 35)

    def test_reward_is_the_sum_of_terms(self) -> None:
        self.assertAlmostEqual(rollout().reward(), 0.75)

    def test_trainability_requires_a_mask_of_the_right_length(self) -> None:
        self.assertTrue(rollout().is_trainable())
        self.assertFalse(rollout(mask=None).is_trainable())
        self.assertFalse(rollout(mask=[1, 1, 1]).is_trainable())

    def test_round_trip_keeps_every_field(self) -> None:
        row = rollout(extras={"note": "kept"})
        again = Rollout.from_json(row.to_json())
        self.assertEqual(again.to_dict(), row.to_dict())

    def test_json_is_ascii_and_sorted(self) -> None:
        text = rollout().to_json()
        text.encode("ascii")  # raises if a non-ASCII character slipped in
        self.assertEqual(text, json.dumps(json.loads(text), ensure_ascii=True, sort_keys=True))

    def test_reading_skips_blank_lines_and_reports_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "rows.jsonl"
            write_rollouts(str(good), [rollout()])
            good.write_text(good.read_text() + "\n\n", encoding="utf-8")
            self.assertEqual(len(read_rollouts(str(good))), 1)

            bad = Path(tmp) / "bad.jsonl"
            bad.write_text('{"episode_id": "e2"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_rollouts(str(bad))
            self.assertIn("bad.jsonl:1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
