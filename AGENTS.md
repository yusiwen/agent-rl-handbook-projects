# AGENTS.md - rules for this repository

Companion code to a book. The book's rules apply here too, because a repo that contradicts its
own book teaches the wrong thing.

## Non-negotiables

1. **English everywhere.** Every file - code, comments, docstrings, commit messages, configs -
   is English. The character check in `scripts/verify.sh` rejects non-English characters the same
   way the book's gate does.
2. **The contract layer stays dependency-free.** `common/` may import only the standard library.
   It must never import torch, transformers, trl, verl, vllm or anything with a GPU in it. The
   whole point is that six projects share one definition of a rollout; a shared definition that
   drags a framework in cannot be shared.
3. **No dataset payloads in git.** No parquet, csv, jsonl, arrow and no downloaded weights under
   `datasets/`. Only pull scripts, a manifest, licence notes and row counts. Enforced by the gate.
4. **No secret ever lands in a file.** Keys, tokens and connection strings are read from the
   environment. If one appears in a diff, stop and report it.
5. **Label every number.** `measured` (you ran it here), `reported` (a paper says so), or
   `arithmetic` (derived from stated assumptions). An unlabelled number is a bug.
6. **Never delete a file or a directory without asking.** Same rule as the book repo.

## What belongs where

| Layer | Holds | Must not hold |
|---|---|---|
| `common/schema.py` | the rollout row, the segment, the reward terms | framework objects, tensors |
| `common/mask.py` | loss-mask construction *and* the fault check | the training loop |
| `common/verifier.py` | the `Verifier` protocol, the registry, the coverage report | any concrete dataset checker |
| `common/fusion.py` | merging rule and model terms, conflict detection | model calls |
| `common/runs.py` | run headers, per-round records, sample ids | plotting, dashboards |
| `common/checks.py` | the five assertions the book's chapters promise | project-specific logic |
| `p*/` | one project: harness, configs, fixtures, its own checks | a second copy of the schema |

## The five checks

`common/checks.py` implements the assertions the book makes to the reader:

1. **mask** - the tokens excluded from the loss are exactly the tokens the policy did not produce.
2. **boundaries** - the recorded turn count matches the turn boundaries rebuildable from the trace.
3. **fusion** - reward terms are recorded separately, and a rule/judge conflict is observable.
4. **advantage spread** - most GRPO groups have a non-zero spread (a group with one reward value
   produces no gradient).
5. **round stability** - the same sample ids appear across rounds with different rewards.

These are the graders. A project that fails one of them fails it loudly, naming the layer.

## Workflow

- `bash scripts/verify.sh` before every commit: tests, English-only, no payloads, P1 check.
- One concern per commit, English imperative subject.
- New project directory = README first (the nine-item template from the book's appendix), then
  fixtures, then code. Fixtures before code is not a style preference: it is how the reader can
  tell the environment works before suspecting their own implementation.

## Known traps (paid for already)

- A `Rollout` with `mask=None` is *not* a valid training row. The check reports it rather than
  guessing, because "no mask" and "all-masked-in" look identical downstream.
- `python3 -m unittest discover` needs `-t .` from the repo root, or the `common` package is not
  importable.
- Project directories use hyphens (`p1-verifiers`) so their Python package cannot use the same
  name; the package inside is `p1_verifiers`, and the check script puts `src/` on `sys.path`.
