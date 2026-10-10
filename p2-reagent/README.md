# P2 - process feedback and turn-level credit

**Status: not started.** Written with Chapter 6.
Book chapter: Chapter 6 (credit assignment), project 2 in the book's appendix.
Compute tier: ✅ - a 0.6B to 3B policy is expected to be enough for the shrinking config.

## What you build

A multi-turn agent loop with a **reasoning reward model** in it, and credit that can be
attributed to a turn rather than to an episode:

```
task -> agent loop (tools) -> per-turn feedback -> reward terms -> policy update
          |                        |
          |                        +-- a criticism and a score per turn, not one number
          +-- turn boundaries recorded, because nothing downstream can rebuild them
```

The contract is already fixed by `../common/`:

* every rollout is a `common.schema.Rollout` with `turn_count` and a `mask`
  (`../common/mask.py` explains why a tool token must never be trained on);
* feedback arrives as **named terms** (`../common/fusion.py`), so "the rule said yes and
  the critic said no" stays readable;
* the run is checked by `../common/checks.py` before any curve is believed.

## Why this project exists

An episode-level reward says "the task failed". It does not say which of five turns went
wrong, and the gradient it produces is spread across all of them. This project is where
that stops being an abstraction.

## What is not here yet

- The environment: use ALFWorld or tau-bench (see `../datasets/manifest.yaml`) for the
  first version. Both run on CPU, which keeps the interesting part - attribution - from
  being drowned by a slow simulator.
- The RRM service, its container, and the four-stage schedule (architecture, SFT, reward
  fusion, training) from the chapter.
