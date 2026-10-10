"""Regenerate the fixture rounds used by the tests and by P1's check script.

    python3 common/tests/make_fixtures.py

The fixtures are committed so that a fresh clone can run everything offline.  They are
generated rather than hand-written because a hand-written mask is exactly the kind of
thing that is wrong without looking wrong.

Three files are produced:

``round-000.jsonl``  two GRPO groups, three samples each, agentic traces with tool
                     segments, correct masks, and one deliberate rule/judge conflict.
``round-001.jsonl``  the same sample ids one round later, with rewards that moved.
``bad-mask.jsonl``   the first round with the tool tokens of one sample left trainable.

The generator asserts that the corruption landed.  A negative fixture that is not
actually broken makes the check that reads it meaningless - the failure mode this repo
warns about in ``AGENTS.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.mask import build_mask, mask_faults, turn_boundaries  # noqa: E402
from common.schema import Rollout, Segment, write_rollouts  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

ENV = "fixtures/v1 (synthetic, no tokeniser)"
CONFIG = "sha256:fixtures0000000"
CODE = "git:fixtures"


def agentic_trace(question_len: int = 40, first_action: int = 60, tool_reply: int = 30,
                  final_action: int = 50) -> list[Segment]:
    """A two-turn agentic trace: think, call a tool, read the reply, answer."""
    return [
        Segment(role="system", text="You are a careful assistant.", tokens=20),
        Segment(role="user", text="Q" * question_len, tokens=question_len),
        Segment(role="model", text="Let me check that with the calculator.", tokens=first_action),
        Segment(role="tool", text="O" * tool_reply, tokens=tool_reply),
        Segment(role="model", text="The answer is 42.", tokens=final_action),
    ]


def make_rollout(prompt_id: str, sample_index: int, rule: float, judge: float) -> Rollout:
    segments = agentic_trace()
    rollout = Rollout(
        episode_id=f"{prompt_id}-s{sample_index}",
        prompt_id=prompt_id,
        sample_index=sample_index,
        group_id=prompt_id,
        segments=segments,
        reward_terms={"rule": rule, "judge": judge},
        env_version=ENV,
        config_hash=CONFIG,
        code_sha=CODE,
        mask=build_mask(segments),
        turn_count=len(turn_boundaries(segments)),
    )
    faults = mask_faults(rollout.segments, rollout.mask)
    assert not faults, f"fixture for {rollout.episode_id} has a bad mask: {faults}"
    return rollout


def round_zero() -> list[Rollout]:
    """Two groups.  Group A is mixed; group B contains a rule/judge conflict."""
    return [
        # group A: one clear success, one clear failure, one middling
        make_rollout("p1", 0, rule=1.0, judge=0.2),
        make_rollout("p1", 1, rule=0.0, judge=-0.4),
        make_rollout("p1", 2, rule=0.0, judge=0.1),
        # group B: sample 0 was scored right by the checker and wrong by the judge
        make_rollout("p2", 0, rule=1.0, judge=-0.3),
        make_rollout("p2", 1, rule=1.0, judge=0.1),
        make_rollout("p2", 2, rule=0.0, judge=0.0),
    ]


def round_one() -> list[Rollout]:
    """Same sample ids, rewards moved - what the round-stability check wants to see."""
    return [
        make_rollout("p1", 0, rule=1.0, judge=0.5),
        make_rollout("p1", 1, rule=0.0, judge=-0.2),
        make_rollout("p1", 2, rule=1.0, judge=0.4),
        make_rollout("p2", 0, rule=1.0, judge=0.2),
        make_rollout("p2", 1, rule=1.0, judge=0.1),
        make_rollout("p2", 2, rule=1.0, judge=0.3),
    ]


def broken_mask(rows: list[Rollout]) -> list[Rollout]:
    """Leave the tool reply trainable in the first row, on purpose."""
    broken = rows[0]
    segments = broken.segments
    offset = 0
    mask = list(broken.mask or [])
    for segment in segments:
        if segment.role == "tool":
            for index in range(offset, offset + segment.tokens):
                mask[index] = 1
        offset += segment.tokens
    broken.mask = mask
    faults = mask_faults(broken.segments, broken.mask)
    assert faults, "the negative fixture is not actually broken - fix the generator"
    print(f"  bad-mask fixture carries {len(faults)} fault(s), first: {faults[0]}")
    return rows


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    zero = round_zero()
    one = round_one()
    write_rollouts(str(FIXTURES / "round-000.jsonl"), zero)
    write_rollouts(str(FIXTURES / "round-001.jsonl"), one)
    write_rollouts(str(FIXTURES / "bad-mask.jsonl"), broken_mask(list(round_zero())))
    for name in ("round-000.jsonl", "round-001.jsonl", "bad-mask.jsonl"):
        path = FIXTURES / name
        print(f"{path.name}: {sum(1 for _ in path.open())} row(s), {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
