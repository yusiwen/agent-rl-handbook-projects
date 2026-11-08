# P5 - asynchronous multi-turn training

**Status: measured.** All three stages ran on the Spark, the asynchronous loop exists in two forms
(sync and async), and every number below is recomputed by `checks/p5_claims.py` from the records in
`fixtures/runs/`.
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

The task set is generated here rather than downloaded: nine tasks over five documents with four real
tools (`search`, `open`, `calc`, `submit`), a microsecond reset, and an oracle that solves every task
so a zero reward means the policy failed rather than the fixture being impossible. The earlier plan
was ALFWorld and TextWorld; it was dropped because the measurement is about the loop, and a world
whose reset costs milliseconds makes staleness a property of the filesystem.

## What is measured

| stage | what ran | result |
|---|---|---|
| 1, cold start | 20 oracle traces, 30 steps, LoRA r16 | loss 2.74 to 0.0008 in 47.8 s, peak 22.8 GB; greedy **7 of 9** |
| 2, outcome reward | four loops, 72 samples each, sync and async | non-zero advantage in **0, 2 and 3** samples; evaluation unchanged at 7 of 9 |
| 3, process reward | same loop, step-level credit | **21** samples carried an advantage, and the policy got worse: 3 of 9 solved, 28 tool errors, 5 episodes out of turns |

The reason stage 2 found almost nothing is in the per-task counts and is the chapter's headline: seven
tasks were solved in all eight sampled attempts, and the two failures were deterministic. A task
solved every time and a task failed every time contribute the same amount to an update - nothing.

The systems half of the project is measured separately, and it is the part that does not depend on the
reward: **36.1** samples per minute synchronously with one worker, **35.9** asynchronously with one
worker, **108.2** with three. Asynchrony alone bought nothing here, because the served model was the
bottleneck and not the trainer; the threefold gain came from running three workers, which asynchrony
permits rather than provides. Staleness stayed at 0.04, 0.06 and 0.18 (mean) with a maximum of one
version, so the staleness bound never discarded a sample - a fact about this world being fast, not
about the bound being right.

## What is not here yet

- **A learned PRM.** Stage 3's process reward is a rule (+0.1 per valid call, -0.3 per error), not
  P2's model. The rule is enough to show the failure mode - it teaches tool-call farming - and not
  enough to show what a learned reward model would do.
- **More than one machine.** The rollout workers, the trainer and the served policy share one GPU;
  the staleness numbers above are honest for that arrangement and say nothing about a cluster where
  the trainer is the bottleneck.
- **A world whose failures are convention-shaped.** The two deterministic failures are arithmetic and
  protocol, which is why a procedure had nothing to teach in Project 6 either.
