# agent-rl-handbook-projects

Companion code for the handbook *Agent RL Explained: A Beginner's Handbook*
(<https://yusiwen.cn/agent-rl-handbook/>). Six projects, one shared contract.

**Progress lives in one place:** [`ROADMAP.md`](https://github.com/yusiwen/agent-rl-handbook/blob/main/ROADMAP.md)
in the book repository is generated from both repositories - chapters, projects, tests, run records -
and the book's gate fails when it drifts.

**Status: fields 1 and 2 are built, and P2 has been trained.** The shared layer (`common/`) is
implemented and tested; P1 (verifier suite) runs end to end on fixtures, offline, and against a
served model over HTTP, graded by a check script that was negative-tested. P2 (the credit-assignment
loop) closes its loop offline on a tabular policy, and has now been carried through all four of its
stages on the reference machine: a supervised fine-tune of a 1.7B policy on 117 oracle
demonstrations, the same loop driven by the served checkpoint, task calibration read out of the
recorded rollouts, and a budgeted judge measured against the rule critic and demoted to a diagnostic
because it disagrees with the rule on three quarters of the turns it scored. The update on top of
the served policy is not built - that loop runs frozen, and P2's README says so. P3-P6 are
directories with a README each, written when their chapter is written. Every number in these
documents comes from a run whose record is named beside it.

## What runs today

```bash
uv sync --frozen                # once; direnv does this for you after `direnv allow`
uv run --frozen bash scripts/verify.sh
```

`uv run` puts `.venv/bin` first on `PATH`, so the `python3` the gate calls is the pinned one, not
whatever the machine happens to have.

```
== common tests ==                Ran 128 tests ... OK
== p1 tests ==                    Ran 190 tests ... OK
== p1 check (fixtures) ==         coverage 21/21 tasks; 42 rollouts; 21/42 passed
                                  contamination 0/21 tasks vs gsm8k/test (1319 items, sha256:524c11ab4ee19ae2)
== p2 tests ==                    Ran 87 tests ... OK
== p2 check (loop) ==             [PASS] every task solvable, the loop closes, no entropy cliff
== p4 tests ==                    Ran 24 tests ... OK
== p4 check (loop) ==             six fixtures reproduced by their oracles, through the sandbox
== p4 claims (arithmetic) ==      2 claims, arithmetic and records agree
== datasets tests ==              Ran 39 tests ... OK
== repository checks (selftest) == [PASS] the checks still reject what they must reject
== repository checks ==           [PASS] characters  [PASS] payloads  [PASS] secrets  [PASS] imports
all checks passed
```

468 tests, no network, no GPU, no weights. The fixture run prints a coverage report and
writes a run record to `runs/` (gitignored). The gate itself sets `ARL_RUNS_DIR` to a scratch
directory, so verifying the tree never adds a record to it; set the same variable to send your own
runs elsewhere. If the gate fails, the problem is in this repository rather
than in your environment - which is the whole reason it exists.

## The environment is pinned

`pyproject.toml` + `uv.lock` pin the packages, `.python-version` pins the interpreter, and
`.envrc` wires the three to direnv:

- **One dependency for the gate: PyYAML.** The contract layer's only import is the standard
  library, and `scripts/check_repo.py --imports` keeps it that way - an environment that
  cannot import torch cannot quietly start depending on it. `uv sync` therefore installs one
  package, in a `.venv` that is gitignored.
- **The interpreter is pinned, not inherited.** `.python-version` says 3.14; `uv run` puts
  `.venv/bin` first on `PATH`, so the `python3` the gate calls is the same one on both
  machines instead of whatever `python3 -V` happens to say that day.
- `huggingface_hub` is an extra rather than a dependency (`uv sync --extra datasets`), because
  only `datasets/pull.py` needs it and only when it pulls. The tool imports both libraries
  inside its functions, which is why the gate runs offline with neither present. See
  `datasets/README.md`.
- **There is no sandboxed second gate any more.** `nix build` used to run the same checks in a
  derivation with no network and a read-only tree. With nix gone, `uv run --frozen bash
  scripts/verify.sh` is the gate, and `uv.lock` is what makes it reproducible; that is a weaker
  guarantee than a derivation and it is written down here rather than implied away.
- `uv sync` reads `pyproject.toml` and `uv.lock`, not the git index, so an untracked file is
  visible to the gate. Run `git add` before committing anyway - the gate's own repository
  check reads the index, and a file the index does not know about is one it cannot check.

## What this repo is for

The book asks you to build six things. This repo holds the *shapes* they share, so that six
projects do not become six dialects:

| Project | Book chapter | Compute | What you build | Config |
|---|---|---|---|---|
| P1 verifiers | Ch 5 | ✅ | A verifier suite and its coverage report | `p1-verifiers/configs/spark-0.6b.yaml` |
| P2 reagent | Ch 6 | ✅ | A process-feedback loop with turn-level credit | `p2-reagent/configs/...` |
| P3 verl runtime | Ch 9 | ⚠️ | One GRPO run under a framework runtime | `p3-verl/configs/...` |
| P4 deepanalyze | Ch 10 | ⚠️ | A data-analysis agent, end to end (measured: 4 of 6 with the right executor) | `p4-deepanalyze/sandbox/Dockerfile` |
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
python3 -m unittest discover -s datasets/tests -t datasets/tests  # the pull tool, offline
python3 p1-verifiers/checks/check_p1.py                # P1 end to end on fixtures
python3 scripts/check_repo.py --selftest               # prove the checks can still fail
python3 scripts/check_repo.py                          # characters, payloads, secrets, imports
python3 datasets/pull.py --list                        # datasets, and what a pull would record
python3 datasets/pull.py gsm8k                         # pull one (payload ignored, digest committed)
python3 p1-verifiers/checks/run_p1_live.py --base-url http://host:8000 --model the-id
                                                       # the same pipeline against a real model
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
- **English prose, by an allowlist of characters** - the same list as the handbook's gate, so the
  two repositories cannot judge the same sentence two different ways. See `scripts/check_repo.py`.
- **The checks are themselves checked.** `check_repo.py --selftest` proves the scanner still
  rejects what it must; a scan that has quietly stopped failing reports success for everything
  else too.
- **Counts are claims.** Row counts, dataset sizes and timings here are labelled
  measured / reported / arithmetic. See `datasets/README.md`.

## Layout

```
pyproject.toml   the gate's one dependency; uv.lock pins it, .python-version pins the interpreter
common/          schema, mask, verifier protocol, fusion, run records, contamination, checks (+ 126 tests)
p1-verifiers/    src/ fixtures/ checks/ configs/ env/ tests/   (harness, live path, Chapter 7's health report)
p2-reagent/      src/ fixtures/ checks/ tests/   (the credit-assignment loop; the policy is a table)
p3..p6/          one README each, written with their chapter
datasets/        pull scripts + manifest + digests + rows_jsonl.py (JSONL views the stdlib checks can read)
scripts/         verify.sh (the gate), check_repo.py (characters, payloads, secrets, selftest)
```

## Licence

MIT for the code (`LICENSE`). Datasets keep their own licences - see `datasets/README.md`.
