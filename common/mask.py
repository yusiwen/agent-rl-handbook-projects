"""Loss masks: what the policy is trained on, and how to prove it is right.

The rule, from Chapter 2 and repeated in Chapter 6: **the loss is computed only on
tokens the policy itself produced**.  Tool output, user messages and system prompts
are context, not targets.  Train on a tool's tokens and the model learns to imitate
the tool instead of using it.

This module does two things:

* ``build_mask`` - turn a trace into a mask, given the roles.
* ``mask_faults`` - compare a *recorded* mask against the trace and report every
  token where they disagree, by role.

The second function is the point of the module.  A mask that is recomputed at check
time can never be wrong; the mask that matters is the one your pipeline recorded
while it was building the batch.  That is the one this checks.

A turn, for accounting purposes, is one ``model`` segment together with the ``tool``
segment that follows it (if any).  This is the unit Chapter 6 attributes credit to,
and the unit the loss mask must not smear across.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Rollout, Segment

MODEL = "model"
TOOL = "tool"


@dataclass(frozen=True)
class Span:
    """A half-open token range ``[start, end)``."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Turn:
    """One model action and the environment reply that answered it."""

    index: int
    model: Span
    tool: Span | None


# --------------------------------------------------------------------- building


def build_mask(segments: list[Segment]) -> list[int]:
    """Return one 0/1 per token: 1 where the policy wrote it, 0 everywhere else.

    Raises ``ValueError`` on an empty trace, because an empty mask is a silent bug
    in every pipeline that consumes it.
    """
    if not segments:
        raise ValueError("cannot build a mask for an empty trace")
    mask: list[int] = []
    for segment in segments:
        mask.extend([1 if segment.role == MODEL else 0] * segment.tokens)
    return mask


def role_at(segments: list[Segment], index: int) -> str:
    """Which actor produced the token at ``index``."""
    offset = 0
    for segment in segments:
        if index < offset + segment.tokens:
            return segment.role
        offset += segment.tokens
    raise IndexError(f"token index {index} is outside the trace ({offset} tokens)")


def spans_for(segments: list[Segment], role: str) -> list[Span]:
    """Every token range produced by ``role``, in order."""
    spans: list[Span] = []
    offset = 0
    for segment in segments:
        if segment.role == role:
            spans.append(Span(offset, offset + segment.tokens))
        offset += segment.tokens
    return spans


def turn_boundaries(segments: list[Segment]) -> list[Turn]:
    """Rebuild the turns of a trace from its roles.

    A ``system`` or ``user`` segment neither opens nor closes a turn - they are not
    actions.  A ``model`` segment with no following ``tool`` segment still counts as
    a turn: it is the last action of the episode, and it is exactly the case where a
    pipeline is most likely to forget to record a boundary.
    """
    turns: list[Turn] = []
    offset = 0
    open_model: Span | None = None
    for segment in segments:
        start = offset
        offset += segment.tokens
        if segment.role == MODEL:
            if open_model is not None:
                turns.append(Turn(index=len(turns), model=open_model, tool=None))
            open_model = Span(start, offset)
        elif segment.role == TOOL and open_model is not None:
            turns.append(Turn(index=len(turns), model=open_model, tool=Span(start, offset)))
            open_model = None
    if open_model is not None:
        turns.append(Turn(index=len(turns), model=open_model, tool=None))
    return turns


# ------------------------------------------------------------------- checking


def mask_faults(segments: list[Segment], mask: list[int] | None) -> list[str]:
    """Return a human-readable fault per token where the mask disagrees with the trace.

    An empty list means the mask is right.  ``mask=None`` is itself a fault: a row
    without a mask is not trainable, and "no mask" and "everything masked in" are
    indistinguishable once they reach a trainer.
    """
    total = sum(segment.tokens for segment in segments)
    if mask is None:
        return ["no mask recorded (a row without a mask cannot be trained on)"]
    faults: list[str] = []
    if len(mask) != total:
        faults.append(f"mask has {len(mask)} entries but the trace has {total} tokens")
        return faults
    for index, value in enumerate(mask):
        role = role_at(segments, index)
        expected = 1 if role == MODEL else 0
        if int(value) != expected:
            faults.append(
                f"token {index} belongs to {role!r} but mask={int(value)} (expected {expected})"
            )
    return faults


def check_rollout_mask(rollout: Rollout) -> list[str]:
    """``mask_faults`` with the episode id attached, for use by the checks."""
    return [f"{rollout.episode_id}: {fault}" for fault in mask_faults(rollout.segments, rollout.mask)]


def describe_mask(segments: list[Segment], limit: int = 60) -> str:
    """A compact picture of the mask, for logs: ``1`` trained, ``0`` masked, ``|`` turn edge."""
    mask = build_mask(segments)
    edges = {turn.model.start for turn in turn_boundaries(segments)}
    pieces: list[str] = []
    for index, value in enumerate(mask[:limit]):
        if index in edges and pieces:
            pieces.append("|")
        pieces.append(str(value))
    suffix = "..." if len(mask) > limit else ""
    trained = sum(mask)
    return f"{''.join(pieces)}{suffix}  ({trained}/{len(mask)} tokens trained)"
