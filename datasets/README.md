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
datasets/manifest.yaml        every dataset: id, licence, row counts, patterns, used by
datasets/pull.py              the pull tool: resolve the revision, fetch, attest, record
datasets/gsm8k.pull.sh        a one-line wrapper for `python3 datasets/pull.py gsm8k`
datasets/digests/<name>.json  committed: which revision, which files, which hashes
datasets/payloads/            ignored; where the pulled files land
```

`huggingface_hub` and PyYAML come from the dev shell's pin rather than from pip, so what a
pull records does not depend on when somebody last ran an installer. The tool imports them
inside the functions that use them, which is why the gate - which must work offline - never
touches either.

The digest is what a run record cites, so somebody else can tell whether they are running
your experiment or a similar-looking one. It also records `allow_patterns`, and refuses to
be written when the payload directory holds files those patterns do not match: a download
adds to a directory rather than replacing it, so an earlier wider pull can leave a filtered
pull looking complete.

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
   into a fixture without attribution. If the card states no licence at all, write
   `unstated` and record the decision you made - "unstated" is not permission.
2. Add a manifest entry: the date you checked, the counts you actually saw, and
   `allow_patterns` if you only want part of it. **No new code is needed** - `pull.py` reads
   the manifest, so a dataset is added by describing it. A per-dataset shell script would be
   a second place for the licence and the patterns to be wrong.
3. Pull it: `python3 datasets/pull.py <name>`. Commit the digest under `datasets/digests/`;
   never the payload. If it refuses because the directory holds files outside the patterns,
   that is the check doing its job - pull into an empty directory, or pass `--clean` and read
   what it says it removed.
4. Record the subset you will use for the reference machine. "All of it" is almost never
   the right answer: the shrink knob that matters most is the task count, because it is
   multiplied by samples per prompt and by turns.
