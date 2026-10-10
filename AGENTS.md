# AGENTS.md - rules for this repository

Companion code to a book. The book's rules apply here too, because a repo that contradicts its
own book teaches the wrong thing.

## Non-negotiables

1. **English everywhere.** Every file - code, comments, docstrings, commit messages, configs -
   is English. The rule is about *characters*, not about a language: `scripts/check_repo.py`
   allows ASCII plus an explicit four-codepoint allowlist (the tier marks in the README), and
   rejects everything else by file, line and codepoint. An allowlist, not a range - a range
   would quietly permit the next full-width accident.
2. **The contract layer stays dependency-free.** `common/` may import only the standard library.
   It must never import torch, transformers, trl, verl, vllm or anything with a GPU in it. The
   whole point is that six projects share one definition of a rollout; a shared definition that
   drags a framework in cannot be shared.
3. **No dataset payloads in git.** No parquet, csv, jsonl, arrow and no downloaded weights.
   `datasets/` holds pull scripts, the manifest, licence notes, row counts - and
   `datasets/digests/`, which *is* committed: a digest is provenance, and it is what a run
   record cites. Data lands in `datasets/payloads/`, which is ignored. Enforced by the gate,
   which reads the git index - so `git add` before running it.
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

## The rules this repository keeps

- **A missing verifier declines; it never scores zero.** An unimplemented checker must return
  `False` from `can_verify` and raise from `verify`. The dangerous version of "not implemented"
  is the one that returns `0.0`, because the training run stays green while the model learns
  nothing - and the coverage report is the number that makes it visible.
- **A verifier and an extractor are two things, and the row records which one failed.** Comparing
  a whole response against an answer scores everything zero; extracting "the last line" then
  comparing is the real shape, and both halves have their own failure mode (`150 minutes` for
  `150`, `x = 12` for `12`). The book's Chapter 5 lesson applies to this repository first.
- **A negative test asserts that the fault landed.** `make_fixtures.py` checks that
  `bad-mask.jsonl` actually carries its mask faults; a corruption that silently failed to apply
  turns a passing test into a decoration.
- **A test that exercises a *failing* fixture must silence the library that prints.** Otherwise a
  thirty-line failure lands in the gate's log and reads exactly like the gate itself failing.
  `common/tests/test_checks.py` wraps those calls in `quiet()`.
- **Fixtures are made up but their contract is asserted.** P1's fixtures declare a pass rate and
  a mixed-group count, and the check script fails if they drift, because a fixture whose contract
  is unstated drifts into being convenient.

## The five checks

`common/checks.py` implements the assertions the book makes to the reader:

1. **mask** - the tokens excluded from the loss are exactly the tokens the policy did not produce.
2. **boundaries** - the recorded turn count matches the turn boundaries rebuildable from the trace.
3. **terms** - reward terms are recorded separately, and a rule/judge conflict is observable.
4. **advantage spread** - most GRPO groups have a non-zero spread (a group with one reward value
   produces no gradient).
5. **round stability** - the same sample ids appear across rounds with different rewards.

These are the graders. A project that fails one of them fails it loudly, naming the layer.

## Workflow

- `bash scripts/verify.sh` before every commit: the two test suites, the P1 fixture run, then
  `scripts/check_repo.py` (characters, payloads, secrets). Exit status is non-zero on any failure,
  so it is a CI step unchanged.
- One concern per commit, English imperative subject.
- New project directory = README first (the nine-item template from the book's appendix), then
  fixtures, then code. Fixtures before code is not a style preference: it is how the reader can
  tell the environment works before suspecting their own implementation.
- P1's own test suite has a different discovery root (`-s p1-verifiers/tests -t p1-verifiers/tests`),
  because the directory's hyphen makes it impossible to import as a package from the repo root.

## Known traps (paid for already)

- A `Rollout` with `mask=None` is *not* a valid training row. The check reports it rather than
  guessing, because "no mask" and "all-masked-in" look identical downstream.
- `python3 -m unittest discover` needs `-t .` from the repo root, or the `common` package is not
  importable.
- Project directories use hyphens (`p1-verifiers`) so their Python package cannot use the same
  name; the package inside is `p1_verifiers`, and the check script puts `src/` on `sys.path`.
- A pull script that hashes everything under its output directory will hash the CLI's own
  `.cache/` bookkeeping too, and a digest that changes when a client updates is not a pin. Skip
  any path whose components start with a dot.
- macOS has no `timeout`; it is `gtimeout` from coreutils. Do not put `timeout` in a script that
  has to run on the reference machine.
