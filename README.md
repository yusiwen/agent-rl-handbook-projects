# agent-rl-projects

Companion code for the handbook *Agent RL Explained: A Beginner's Handbook*
(<https://yusiwen.cn/agent-rl-handbook/>). Six projects, one shared contract.

**Status: field 1 is built.** The shared layer (`common/`) is implemented and tested; P1
(verifier suite) runs end to end on fixtures, offline, and is graded by a check script that
was negative-tested. P2-P6 are directories with a README each, written when their chapter is
written. Nothing here has been trained yet, and nothing here claims a number it did not
measure.

## What runs today

```bash
bash scripts/verify.sh
```

```
== common tests ==        Ran 97 tests ... OK
== p1 tests ==            Ran 36 tests ... OK
== p1 check (fixtures) == coverage 10/11 tasks; 20 rollouts; 14/20 passed, 6/10 groups mixed
== repository checks ==   [PASS] characters   [PASS] payloads   [PASS] secrets
all checks passed
```

133 tests, no network, no GPU, no weights. The fixture run prints a coverage report and
writes a run record to `runs/` (gitignored). If the gate fails, the problem is in this
repository rather than in your environment - which is the whole reason it exists.

## What this repo is for

The book asks you to build six things. This repo holds the *shapes* they share, so that six
projects do not become six dialects:

| Project | Book chapter | Compute | What you build | Config |
|---|---|---|---|---|
| P1 verifiers | Ch 5 | ✅ | A verifier suite and its coverage report | `p1-verifiers/configs/spark-0.6b.yaml` |
| P2 reagent | Ch 6 | ✅ | A process-feedback loop with turn-level credit | `p2-reagent/configs/...` |
| P3 verl runtime | Ch 9 | ⚠️ | One GRPO run under a framework runtime | `p3-verl/configs/...` |
| P4 deepanalyze | Ch 10 | ⚠️ | A data-analysis agent, end to end | `p4-deepanalyze/configs/...` |
| P5 openclaw-rl | Ch 10 | ⚠️ | An asynchronous multi-turn training loop | `p5-openclaw-rl/configs/...` |
| P6 memento-skills | Ch 12 | ✅ | A skill library that improves without training | `p6-memento-skills/configs/...` |

A ✅ config must run on the reference machine (a DGX Spark: 128 GB unified memory). A ⚠️ config
must run *shrunk*; files named `rented-*.yaml` are for a rented GPU and are marked
`read, do not run here`.

## Run it

```bash
bash scripts/verify.sh                                 # the whole gate
python3 -m unittest discover -s common/tests -t .      # the shared layer
python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests
python3 p1-verifiers/checks/check_p1.py                # P1 end to end on fixtures
python3 scripts/check_repo.py                          # characters, payloads, secrets
bash datasets/gsm8k.pull.sh                            # pull a dataset (writes an ignored payload)
```

The gate reads the git index, so run `git add` first if you want a brand-new file checked.

## The rules this repo keeps

- **No data payloads in git.** `datasets/` holds pull scripts, pinned revisions, licence notes
  and row counts; `datasets/digests/` holds what a run must cite. Some of these datasets are
  share-alike (DS-1000 and HotpotQA are CC-BY-SA-4.0) and two declare no licence at all, so
  committing their content would be a licence problem before it was a size problem.
- **The contract layer imports nothing.** `common/` must not import torch, transformers, trl or
  verl. Projects may; the shared shapes may not.
- **Every run is recorded with its environment.** A rollout row carries the environment version,
  the config hash and the code SHA.
- **Reward is never one number.** Rollouts carry *terms*; the fused total is derived. If you
  cannot say how much each term contributed, you cannot debug the run.
- **Counts are claims.** Row counts, dataset sizes and timings here are labelled
  measured / reported / arithmetic. See `datasets/README.md`.
- **Prose is English, and the gate enforces it** - see `scripts/check_repo.py` and
  `AGENTS.md` for why the rule is about characters rather than about a language.

## Layout

```
common/          schema, mask, verifier protocol, fusion, run records, checks (+ 97 tests)
p1-verifiers/    src/ fixtures/ checks/ configs/ tests/   (harness complete, three verifiers yours)
p2..p6/          one README each, written with their chapter
datasets/        pull scripts + manifest + digests (verified row counts and licences)
scripts/         verify.sh (the gate), check_repo.py (characters, payloads, secrets)
```

## Licence

MIT for the code (`LICENSE`). Datasets keep their own licences - see `datasets/README.md`.
