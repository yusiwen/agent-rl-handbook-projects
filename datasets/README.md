# Datasets and environments

**Nothing here is committed except this file, the manifest, and the pull scripts.**
Payloads land in `datasets/payloads/`, which `.gitignore` excludes.

## Why the data is not in the repository

Two reasons, and the second one is the one that bites:

1. **Size.** A task set with its fixtures is a few megabytes; a task set with its
   rollouts is not.
2. **Licence.** Some of the datasets this book needs are share-alike. `xlangai/DS-1000` is
   CC-BY-SA-4.0 and `hotpotqa/hotpot_qa` is CC-BY-SA-4.0. Redistributing them inside an
   MIT-licensed repository would attach their terms to everything downstream. Pull them
   per machine, keep the licence note, and let the manifest record which revision was used.

Two of the most useful sets declare **no licence at all** on their cards
(`HuggingFaceH4/MATH-500`, `WildEval/ZebraLogic`). "No licence" is not "free to use": treat
those as *to check before publishing a derivative*, and record what you decided.

## What is committed

```
datasets/manifest.yaml      every dataset: id, licence, row counts, revision, used by
datasets/*.pull.sh          one script per dataset, pinned to a revision, writing a digest
datasets/payloads/          ignored; where the pulled files land
```

The pull scripts write a `payloads/<name>.digest` next to the data. A run record cites that
digest, which is what lets somebody else tell whether they are running your experiment or a
similar-looking one.

## Verified numbers

Row counts and licences below were read from the Hugging Face API and its datasets-server
on **2026-10-09**. They are `reported`, not `measured`: this project has not trained on
them yet. Anything that changes upstream will be wrong here, which is why the manifest
carries the date and why the pull scripts pin a revision.

| Dataset | Licence | Rows | Used by |
|---|---|---|---|
| `openai/gsm8k` | MIT | train 7,473 / test 1,319 | P1, P3 |
| `HuggingFaceH4/MATH-500` | **unstated** | test 500 | P1, P3 |
| `WildEval/ZebraLogic` | **unstated** | 1,000 / 3,259 (two configs) | P1 |
| `google-research-datasets/mbpp` | CC-BY-4.0 | 374 + 500 + 90 + 10 = 974 | P1 |
| `xlangai/DS-1000` | **CC-BY-SA-4.0** | test 1,000 | P4 |
| `stanfordnlp/wikitablequestions` | CC-BY-4.0 | size API needs a config; not fetched | P4 |
| `wenhu/tab_fact` | CC-BY-4.0 | size API needs a config; not fetched | P4 |
| `hotpotqa/hotpot_qa` | **CC-BY-SA-4.0** | train 90,447 / validation 7,405 | P2 |
| `DigitalLearningGmbH/MATH-lighteval` | MIT | two configs: 7,500/5,000 and 1,744/1,187 | P1, P3 |
| `open-r1/OpenR1-Math-220k` | Apache-2.0 | three configs: 93,733 / 225,129 / 131,396 | P1, P3 |

Corrections worth keeping, because they were wrong in the plan first:

* `hendrycks/competition_math` (the obvious MATH entry) now returns **401**. The canonical
  replacement is `DigitalLearningGmbH/MATH-lighteval`; MATH-500 is the 500-item subset.
* `wikitq/wikitablequestions` does not resolve; the real one is
  `stanfordnlp/wikitablequestions`.
* `xlangai/DS-1000` exists; `DS-1000/DS-1000` returns 401.

## Environments are not datasets

P2, P5 and P6 need *environments*, which are Python packages plus a task list, never a
CSV: ALFWorld, TextWorld, tau-bench and the Berkeley function-calling leaderboard are all
installed and driven by code. They are pinned in the project's own environment file, not
here, because their "row count" is a moving target and their tasks are produced at run
time. What this repository owes them is the *task manifest* format for each project: a
JSONL row with an id, a task specification, and the expected outcome - the same shape
`p1-verifiers/fixtures/tasks.jsonl` uses.

## How to add a dataset

1. Read the licence. If it is share-alike, say so in the manifest and never copy a row
   into a fixture without attribution.
2. Write `datasets/<name>.pull.sh` that pins a revision, writes into `payloads/`, and
   records a digest.
3. Add a manifest entry with the date you checked and the counts you actually saw.
4. Record the subset you will use for the reference machine. "All of it" is almost never
   the right answer: the shrink knob that matters most is the task count, because it is
   multiplied by samples per prompt and by turns.
