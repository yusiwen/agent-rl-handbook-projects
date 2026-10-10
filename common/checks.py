"""The five assertions the book makes to the reader, as code.

A "grader" that only prints a number teaches nothing.  These checks name the layer
that failed - mask, boundaries, terms, spread, rounds - because that is the diagnosis
the book's chapters train you to make.

Run them from the command line::

    python3 -m common.checks --rollouts runs/<id>/round-000.jsonl
    python3 -m common.checks --rollouts round-001.jsonl --previous round-000.jsonl

Exit status is 0 only when every check passes, so this is usable as a CI step.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Sequence

from . import runs
from .mask import check_rollout_mask, turn_boundaries
from .schema import Rollout, read_rollouts

#: Below this, assume a flat group rather than a lucky float difference.
SPREAD_EPSILON = runs.SPREAD_EPSILON

#: The share of GRPO groups that must be able to produce a gradient.  Not 100%: a few
#: all-pass or all-fail prompts are expected in any real task set.  What is not
#: acceptable is a batch where most groups are flat.
DEFAULT_MIN_SPREAD_FRACTION = 0.5


@dataclass
class CheckResult:
    """One assertion's verdict, with the messages a human would need to act on it."""

    name: str
    what_it_checks: str
    ok: bool
    messages: list[str] = field(default_factory=list)

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        head = f"[{status}] {self.name}: {self.what_it_checks}"
        if not self.messages:
            return head
        return "\n".join([head] + [f"       {message}" for message in self.messages])


# ------------------------------------------------------------------- the checks


def check_mask(rollouts: Sequence[Rollout]) -> CheckResult:
    """Every token excluded from the loss was produced by something other than the policy."""
    messages: list[str] = []
    for rollout in rollouts:
        messages.extend(check_rollout_mask(rollout))
    return CheckResult(
        name="mask",
        what_it_checks="the loss mask covers exactly the tokens the policy produced",
        ok=not messages,
        messages=messages,
    )


def check_boundaries(rollouts: Sequence[Rollout]) -> CheckResult:
    """The recorded turn count matches the turns rebuildable from the trace."""
    messages: list[str] = []
    for rollout in rollouts:
        computed = len(turn_boundaries(rollout.segments))
        if rollout.turn_count is None:
            messages.append(f"{rollout.episode_id}: no turn_count recorded (computed {computed})")
        elif rollout.turn_count != computed:
            messages.append(
                f"{rollout.episode_id}: recorded turn_count={rollout.turn_count} "
                f"but the trace has {computed} turn(s)"
            )
    return CheckResult(
        name="boundaries",
        what_it_checks="recorded turn boundaries match the trace",
        ok=not messages,
        messages=messages,
    )


def check_terms(rollouts: Sequence[Rollout]) -> CheckResult:
    """Reward arrives as named terms, so each source's contribution is auditable."""
    messages: list[str] = []
    inventory: set[str] = set()
    for rollout in rollouts:
        if not rollout.reward_terms:
            messages.append(f"{rollout.episode_id}: no reward terms recorded")
        inventory.update(rollout.reward_terms)
    if not messages and not inventory:
        messages.append("no reward terms anywhere in this batch")
    return CheckResult(
        name="terms",
        what_it_checks="every rollout records its reward as named terms",
        ok=not messages,
        messages=messages + ([f"terms seen: {', '.join(sorted(inventory))}"] if inventory else []),
    )


def check_advantage_spread(
    rollouts: Sequence[Rollout],
    min_fraction: float = DEFAULT_MIN_SPREAD_FRACTION,
) -> CheckResult:
    """Most GRPO groups contain more than one distinct reward.

    A group whose attempts all scored the same contributes a zero advantage: the
    gradient has nothing to point at, and training quietly does nothing.
    """
    spreads = runs.spread_per_group(rollouts)
    if not spreads:
        return CheckResult(
            name="advantage spread",
            what_it_checks="most groups can produce a non-zero advantage",
            ok=False,
            messages=["no groups in this batch"],
        )
    usable = [group for group, value in spreads.items() if value > SPREAD_EPSILON]
    flat = sorted(group for group, value in spreads.items() if value <= SPREAD_EPSILON)
    fraction = len(usable) / len(spreads)
    ok = fraction >= min_fraction
    messages = [f"{len(usable)}/{len(spreads)} groups usable ({fraction:.0%})"]
    if flat:
        shown = ", ".join(flat[:5]) + (" ..." if len(flat) > 5 else "")
        messages.append(f"flat groups: {shown}")
    return CheckResult(
        name="advantage spread",
        what_it_checks="most groups can produce a non-zero advantage",
        ok=ok,
        messages=messages,
    )


def check_round_stability(previous: Sequence[Rollout], current: Sequence[Rollout]) -> CheckResult:
    """The same attempts reappear in the next round, with different rewards.

    Two failures are possible here and they mean different things: missing ids means the
    batches are not comparable, and unchanged rewards mean the feedback you added was
    never consumed.
    """
    previous_ids = set(runs.sample_ids(previous))
    current_ids = set(runs.sample_ids(current))
    messages: list[str] = []

    vanished = sorted(previous_ids - current_ids)
    appeared = sorted(current_ids - previous_ids)
    if vanished:
        messages.append(f"missing in the later round: {', '.join(vanished[:5])}"
                        + (" ..." if len(vanished) > 5 else ""))
    if appeared:
        messages.append(f"new in the later round: {', '.join(appeared[:5])}"
                        + (" ..." if len(appeared) > 5 else ""))

    by_id_previous = {runs.sample_id(rollout): rollout for rollout in previous}
    by_id_current = {runs.sample_id(rollout): rollout for rollout in current}
    shared = sorted(previous_ids & current_ids)
    moved = [
        sample
        for sample in shared
        if abs(by_id_current[sample].reward() - by_id_previous[sample].reward()) > SPREAD_EPSILON
    ]
    if shared and not moved:
        messages.append(
            "no shared sample changed its reward between rounds - the feedback is not "
            "reaching the batch"
        )
    elif shared:
        messages.append(f"{len(moved)}/{len(shared)} shared samples changed reward")

    return CheckResult(
        name="round stability",
        what_it_checks="the same samples reappear next round, with changed rewards",
        ok=not vanished and not appeared and bool(moved),
        messages=messages,
    )


def run_all(
    rollouts: Sequence[Rollout],
    previous: Sequence[Rollout] | None = None,
    min_spread_fraction: float = DEFAULT_MIN_SPREAD_FRACTION,
) -> list[CheckResult]:
    """Every check, in the order the book introduces them."""
    results = [
        check_mask(rollouts),
        check_boundaries(rollouts),
        check_terms(rollouts),
        check_advantage_spread(rollouts, min_fraction=min_spread_fraction),
    ]
    if previous is not None:
        results.append(check_round_stability(previous, rollouts))
    return results


# ----------------------------------------------------------------------- the CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m common.checks",
        description="Check a round of rollouts against the book's five assertions.",
    )
    parser.add_argument("--rollouts", required=True, help="JSONL file of rollout rows")
    parser.add_argument("--previous", help="an earlier round, to check round stability")
    parser.add_argument(
        "--min-spread",
        type=float,
        default=DEFAULT_MIN_SPREAD_FRACTION,
        help=f"minimum share of usable groups (default {DEFAULT_MIN_SPREAD_FRACTION})",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    rollouts = read_rollouts(args.rollouts)
    previous = read_rollouts(args.previous) if args.previous else None
    results = run_all(rollouts, previous=previous, min_spread_fraction=args.min_spread)

    if args.json:
        payload = [
            {"name": r.name, "ok": r.ok, "messages": r.messages, "what": r.what_it_checks}
            for r in results
        ]
        print(json.dumps({"rollouts": len(rollouts), "checks": payload}, indent=2))
    else:
        print(f"{len(rollouts)} row(s) from {args.rollouts}")
        for result in results:
            print(result.render())
        failed = [result.name for result in results if not result.ok]
        print("all checks passed" if not failed else f"failed: {', '.join(failed)}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
