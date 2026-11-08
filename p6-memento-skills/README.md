# P6 - a skill library that improves without touching the weights

**Status: in progress.** The task set, the gate and the retrieval measurement exist and are
checked; the outcome half (does the retrieved skill change what the model does) needs the served
policy and is not measured yet. Written with Chapter 12.
Book chapter: Chapter 12 (skills and self-improvement), project 6 in the book's appendix.
Compute tier: ✅ - a small local model plus a retrieval index; no training at all.

## What you build

A loop that turns repeated failure into a reusable skill, and proves it did:

```
task family -> attempt -> failure trace -> candidate skill -> validation -> library
      ^                                                                        |
      +----------------------- next attempt retrieves it ----------------------+
```

## Why it is measured and not asserted

This project is where the book's "a number needs a label" rule earns its keep. Two
measurements have to exist before any claim of self-improvement is meaningful:

* **retrieval**: does the right skill get found? Report it the way the chapter does -
  recall at 1, against a plain keyword baseline and against an embedding baseline. Without
  a baseline, "our router works" is an opinion.
* **outcome**: does the retrieved skill change what happens on the task? Report success
  rate on the *same* task families, with and without the library populated.

Both numbers are `measured`. The baseline is `measured` too. Anything read off a paper is
`reported` and stays labelled that way.

## The task set is generated here, not downloaded

`datasets/manifest.yaml` lists it under `derived`: 10 recurring task families with 20
instances each. The recurrence is the point - a skill library is only interesting when the
same shape of problem comes back, and no public dataset is built to make that happen on
purpose.

## The four questions this project answers

1. When does a skill belong in the library rather than in the prompt?
2. What does a validation step have to be, so that a bad skill cannot get in?
3. What happens to retrieval when there are a hundred skills rather than five?
4. What is the *evidence* that the library helped - and what would have looked like
   evidence while being nothing of the kind?

## What is measured so far: retrieval, at three library sizes

`env/measure_retrieval.py` runs 60 queries (six instances per family) through four routers, with the
library padded to ten, fifty and a hundred documents. The padding is the experiment: at fifty and a
hundred the extra documents are **ten on-topic decoys** - each a plausible procedure for the same
task that fixes the wrong thing - plus word-salad fillers.

| library | documents | keyword | vector | hybrid | idf |
|---|---|---|---|---|---|
| ten | 10 | 0.900 | 0.700 | 1.000 | 1.000 |
| fifty | 50 | 0.600 | 0.367 | 0.400 | **0.800** |
| hundred | 100 | 0.600 | 0.367 | 0.400 | **0.800** |

Three things are visible in that table and each one is a claim the chapter makes:

* **The size of the library is not the problem.** Fifty and a hundred documents score identically,
  so what hurts is *similarity*, not count - ten adjacent skills cost more than eighty unrelated
  ones.
* **A router that matches on the task's own words is matching the wrong thing.** The decoys repeat
  the task vocabulary; the true skill names a gotcha. Jaccard keyword overlap falls to 0.600.
* **Rarity is the cheapest signal that separates a procedure from a restatement of the task.** The
  `idf` router (document frequency over the library, plus a share of the failure symptom) is the
  only one that holds at 0.800 - and it was added *after* the first three collapsed, which the
  record and the chapter both say out loud.

The `hybrid` router is the instructive failure: it had the failure symptom available and still
scored 0.400, because a long procedure dilutes a Jaccard score while a short decoy concentrates it.
Having the right input is not the same as weighting it correctly.

## What is not here yet

- **The outcome half.** A run of `policy(ask, "")` against `policy(ask, skill)` on the same
  held-out instances, using the served model. The gate's logic is tested with a scripted policy
  (`tests/test_p6.py`, `GateTests`); the model run is what turns it into a measurement.
- The failure-trace extractor: `SYMPTOMS` is currently written by hand, one sentence per family.
