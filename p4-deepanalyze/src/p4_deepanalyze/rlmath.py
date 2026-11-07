"""The arithmetic of the P4 reinforcement step, kept away from torch so the gate can test it.

The reward is deliberately small and blunt: the executor decides.  A run either produced the
number the fixture expects (1.0), produced a program that ran but answered something else (0.2),
or produced nothing usable - a crash, a protocol error, a timeout (0.0).  Nothing here inspects
the program's text, its style or its length; if you want the model to write a particular kind of
program, that belongs in the demonstrations, not in the reward.

The advantage is a group mean subtracted from each sample, which is the whole of GRPO's idea
applied at the smallest scale that still means something.  When every sample in a group scores the
same - the common case once a policy is cold-started on six tasks - every advantage is zero, and
the honest reading is that this task contributed no gradient rather than that the update was small.
"""

from __future__ import annotations


def reward_for(solved: bool, ran_ok: bool) -> float:
    """One number per rollout.  `ran_ok` means at least one program exited 0."""
    if solved:
        return 1.0
    return 0.2 if ran_ok else 0.0


def advantages(rewards: list[float]) -> list[float]:
    """Mean-subtracted, no standard deviation: with four samples the std is noise.

    Dividing by the std (as the GRPO paper does) turns a group of `[0.2, 0.2, 0.2, 0.0]` into
    advantages larger than a group of `[1.0, 0.2, 0.0, 0.0]`, which is the opposite of what the
    rewards say.  Dividing by max(std, epsilon) does not fix it; it only hides it.  So the group
    mean is the baseline and nothing else.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [round(value - mean, 6) for value in rewards]


def group_report(groups: dict[str, list[float]]) -> dict[str, object]:
    """Per task: how many samples, how many distinct outcomes, and whether it can teach anything."""
    report: dict[str, object] = {}
    informative = 0
    for task_id, rewards in groups.items():
        spread = max(rewards) - min(rewards) if rewards else 0.0
        teaches = spread > 0
        informative += int(teaches)
        report[task_id] = {
            "samples": len(rewards),
            "solved": sum(1 for r in rewards if r >= 1.0),
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "spread": round(spread, 6),
            "teaches": teaches,
        }
    report["_summary"] = {
        "groups": len(groups),
        "informative_groups": informative,
        "uninformative_groups": len(groups) - informative,
    }
    return report
