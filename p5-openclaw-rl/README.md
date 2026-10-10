# P5 - asynchronous multi-turn training

**Status: not started.** Written with Chapter 10.
Book chapter: Chapter 10 (multi-turn, tool-calling RL), project 5 in the book's appendix.
Compute tier: ⚠️ - a 0.6B policy through all three stages; the asynchronous loop is the
exercise, not the model size.

## What you build

The loop from the book's Chapter 10, at a size that fits one machine:

```
stage 1  supervised fine-tuning on tool-calling traces
stage 2  RL on tool calls, outcome reward only        (ReTool-style)
stage 3  RL with a process reward on the same traces  (adds the PRM)

asynchronous form:
  serving  ->  rollout  ->  reward / judge  ->  policy update
     ^                                              |
     +---------------- weight sync -----------------+
```

## The three things this project is actually about

1. **Staleness must be bounded and recorded.** Once rollout and training overlap, a batch
   is trained on a policy that is no longer the current one. The run record has to say how
   far behind the samples were, or the loss curve is uninterpretable.
2. **The same sample ids must be comparable across stages.** `../common/runs.py` writes
   them for exactly this reason: if stage 3's rewards cannot be lined up against stage 2's,
   nothing can be attributed to the PRM.
3. **A quiet loop looks exactly like a working one.** The checks catch the three silent
   stalls - flat groups, masked-out gradients, and rewards that never move between rounds.

## Environment

ALFWorld and TextWorld first (CPU, fast reset, repeatable - see
`../datasets/manifest.yaml`). A container-based task set is more realistic and much harder
to reset; Chapter 8 is where that trade is discussed, and this project is where it is paid.

## What is not here yet

- The three stage configs and the asynchronous runner.
- A PRM: stage 3 needs a reward model that scores partial traces, which is P2's artefact
  reused rather than a new one.
