"""The accounting of an asynchronous loop: how far behind the samples were, and what that costs.

This module exists because these are the numbers that make an async run interpretable, and because
they are arithmetic: they can be tested without a GPU, a server or a model.  Three things are
computed here and nowhere else:

* **staleness** - the policy has advanced k versions since a sample was generated; k = 0 is fresh.
  A batch's staleness is recorded per sample, and its mean is the honest description of the run;
* **who gets dropped** - a sample beyond the bound is dropped *and counted*.  A loop that quietly
  discards stale samples looks identical to one that never had any;
* **throughput** - samples per minute, which is the only number async mode is trying to improve.
"""

from __future__ import annotations

from statistics import mean


def version_index(name: str) -> int:
    """`p5-v7` -> 7.  Names that do not follow the scheme are version 0, not a crash."""
    _, _, tail = name.rpartition("v")
    try:
        return int(tail)
    except ValueError:
        return 0


def staleness(sample_versions: list[str], current: str) -> list[int]:
    """Per sample: how many policy versions old it is.  Negative values are clamped to 0."""
    now = version_index(current)
    return [max(0, now - version_index(version)) for version in sample_versions]


def partition(batch: list[dict], current: str, bound: int) -> tuple[list[dict], list[dict]]:
    """Split a batch into what may be trained on and what is too old, keeping the ages attached."""
    ages = staleness([row.get("version", "") for row in batch], current)
    keep, drop = [], []
    for row, age in zip(batch, ages):
        row["staleness"] = age
        (keep if age <= bound else drop).append(row)
    return keep, drop


def group_advantages(rewards: list[float]) -> list[float]:
    """Mean-subtracted.  No std division: with four samples it inverts the ordering of small groups."""
    if not rewards:
        return []
    centre = mean(rewards)
    return [round(value - centre, 6) for value in rewards]


def summarise(records: list[dict], seconds: float) -> dict:
    """What the run's own log says, computed from the log rather than from the console."""
    if not records:
        return {"rounds": 0, "samples": 0, "mean_staleness": 0.0, "max_staleness": 0, "dropped": 0}
    samples = sum(row["trained_on"] + row.get("dropped", 0) for row in records)
    return {
        "rounds": len(records),
        "samples": samples,
        "trained_on": sum(row["trained_on"] for row in records),
        "used": sum(row.get("used", 0) for row in records),
        "dropped": sum(row.get("dropped", 0) for row in records),
        "mean_staleness": round(mean(row["mean_staleness"] for row in records), 2),
        "max_staleness": max(row["max_staleness"] for row in records),
        "samples_per_minute": round(samples / max(seconds / 60.0, 1e-6), 1),
        "seconds": round(seconds, 1),
    }
