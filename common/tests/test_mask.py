"""Loss masks: construction, turn boundaries, and the fault report."""

from __future__ import annotations

import unittest

from common.mask import (
    Span,
    build_mask,
    check_rollout_mask,
    describe_mask,
    mask_faults,
    role_at,
    spans_for,
    turn_boundaries,
)
from common.schema import Segment


def trace() -> list[Segment]:
    return [
        Segment(role="system", text="sys", tokens=5),
        Segment(role="user", text="hi", tokens=10),
        Segment(role="model", text="thinking", tokens=20),
        Segment(role="tool", text="42", tokens=5),
        Segment(role="model", text="the answer is 42", tokens=15),
    ]


class SpanTests(unittest.TestCase):
    def test_length_and_overlap(self) -> None:
        self.assertEqual(Span(10, 20).length, 10)
        self.assertTrue(Span(0, 10).overlaps(Span(9, 20)))
        self.assertFalse(Span(0, 10).overlaps(Span(10, 20)))


class BuildMaskTests(unittest.TestCase):
    def test_only_model_tokens_are_trained(self) -> None:
        mask = build_mask(trace())
        self.assertEqual(len(mask), 55)
        self.assertEqual(sum(mask), 35)
        self.assertEqual(mask[:15], [0] * 15)
        self.assertEqual(mask[15:35], [1] * 20)
        self.assertEqual(mask[35:40], [0] * 5)
        self.assertEqual(mask[40:], [1] * 15)

    def test_empty_trace_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build_mask([])

    def test_zero_token_segments_produce_no_entries(self) -> None:
        mask = build_mask([Segment(role="user", text="", tokens=0),
                           Segment(role="model", text="x", tokens=2)])
        self.assertEqual(mask, [1, 1])


class RoleLookupTests(unittest.TestCase):
    def test_role_at_covers_every_token(self) -> None:
        segments = trace()
        self.assertEqual(role_at(segments, 0), "system")   # first token
        self.assertEqual(role_at(segments, 4), "system")   # last system token
        self.assertEqual(role_at(segments, 5), "user")     # first user token
        self.assertEqual(role_at(segments, 14), "user")    # last user token
        self.assertEqual(role_at(segments, 15), "model")   # first token the policy wrote
        self.assertEqual(role_at(segments, 34), "model")   # last token of turn 1's action
        self.assertEqual(role_at(segments, 35), "tool")    # first token of the reply
        self.assertEqual(role_at(segments, 54), "model")   # last token of the episode

    def test_role_at_past_the_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            role_at(trace(), 55)

    def test_spans_for_role(self) -> None:
        self.assertEqual(spans_for(trace(), "model"), [Span(15, 35), Span(40, 55)])
        self.assertEqual(spans_for(trace(), "tool"), [Span(35, 40)])
        self.assertEqual(spans_for(trace(), "system"), [Span(0, 5)])


class TurnBoundaryTests(unittest.TestCase):
    def test_two_turns_and_the_last_one_has_no_tool_reply(self) -> None:
        turns = turn_boundaries(trace())
        self.assertEqual(len(turns), 2)
        self.assertEqual((turns[0].index, turns[0].model, turns[0].tool), (0, Span(15, 35), Span(35, 40)))
        self.assertEqual(turns[1].model, Span(40, 55))
        self.assertIsNone(turns[1].tool)

    def test_system_and_user_segments_do_not_open_turns(self) -> None:
        turns = turn_boundaries(trace())
        self.assertNotIn(Span(0, 5), [turn.model for turn in turns])

    def test_consecutive_model_segments_each_count_as_a_turn(self) -> None:
        segments = [
            Segment(role="model", text="a", tokens=3),
            Segment(role="model", text="b", tokens=4),
            Segment(role="tool", text="c", tokens=2),
        ]
        turns = turn_boundaries(segments)
        self.assertEqual(len(turns), 2)
        self.assertIsNone(turns[0].tool)
        self.assertEqual(turns[1].tool, Span(7, 9))

    def test_a_prompt_with_no_model_segment_has_no_turns(self) -> None:
        self.assertEqual(turn_boundaries([Segment(role="user", text="hi", tokens=4)]), [])


class MaskFaultTests(unittest.TestCase):
    def test_correct_mask_has_no_faults(self) -> None:
        self.assertEqual(mask_faults(trace(), build_mask(trace())), [])

    def test_missing_mask_is_reported_as_such(self) -> None:
        faults = mask_faults(trace(), None)
        self.assertEqual(len(faults), 1)
        self.assertIn("no mask recorded", faults[0])

    def test_wrong_length_is_reported_before_indexing(self) -> None:
        faults = mask_faults(trace(), [1, 1, 1])
        self.assertEqual(len(faults), 1)
        self.assertIn("3 entries", faults[0])
        self.assertIn("55 tokens", faults[0])

    def test_a_trained_tool_token_is_named_with_its_role(self) -> None:
        mask = build_mask(trace())
        mask[36] = 1  # a token inside the tool reply
        faults = mask_faults(trace(), mask)
        self.assertEqual(len(faults), 1)
        self.assertIn("token 36", faults[0])
        self.assertIn("'tool'", faults[0])

    def test_a_masked_model_token_is_caught_too(self) -> None:
        mask = build_mask(trace())
        mask[20] = 0
        faults = mask_faults(trace(), mask)
        self.assertEqual(len(faults), 1)
        self.assertIn("'model'", faults[0])

    def test_the_rollout_helper_prefixes_the_episode_id(self) -> None:
        from common.schema import Rollout

        row = Rollout(
            episode_id="e9",
            prompt_id="p",
            sample_index=0,
            group_id="p",
            segments=trace(),
            reward_terms={"rule": 1.0},
            mask=None,
        )
        faults = check_rollout_mask(row)
        self.assertTrue(faults and faults[0].startswith("e9:"))


class DescribeMaskTests(unittest.TestCase):
    def test_picture_reports_the_trained_share(self) -> None:
        text = describe_mask(trace())
        self.assertIn("35/55 tokens trained", text)
        self.assertIn("|", text)  # turn edges are marked

    def test_long_traces_are_truncated(self) -> None:
        segments = [Segment(role="model", text="x", tokens=500)]
        self.assertTrue(describe_mask(segments, limit=10).endswith("(500/500 tokens trained)"))
        self.assertIn("...", describe_mask(segments, limit=10))


if __name__ == "__main__":
    unittest.main()
