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
nix develop --command bash scripts/verify.sh   # or: direnv allow .  then the same command
nix build --no-link                            # the same gate in a sandbox, no network
```

```
== common tests ==                Ran 126 tests ... OK
== p1 tests ==                    Ran 190 tests ... OK
== p1 check (fixtures) ==         coverage 21/21 tasks; 42 rollouts; 21/42 passed
                                  contamination 0/21 tasks vs gsm8k/test (1319 items, sha256:524c11ab4ee19ae2)
== p2 tests ==                    Ran 81 tests ... OK
== p2 check (loop) ==             [PASS] every task solvable, the loop closes, no entropy cliff
== datasets tests ==              Ran 39 tests ... OK
== repository checks (selftest) == [PASS] the checks still reject what they must reject
== repository checks ==           [PASS] characters  [PASS] payloads  [PASS] secrets  [PASS] imports
all checks passed
```

436 tests, no network, no GPU, no weights. The fixture run prints a coverage report and
writes a run record to `runs/` (gitignored); `nix build` sends it to the sandbox's writable
directory instead, which is what `ARL_RUNS_DIR` is for. If the gate fails, the problem is in
this repository rather than in your environment - which is the whole reason it exists.

## The environment is pinned

`flake.nix` + `flake.lock` pin the toolchain and `.envrc` wires it to direnv:

- **The same nixpkgs revision as the handbook** (`7a0f122f5090`), so both repositories
  resolve to identical store paths on the reference machine instead of downloading a second
  copy of Python.
- The shell is deliberately small - `python3`, `git`, `coreutils`, `ripgrep`, `fd`, `jq`. The
  contract layer's only dependency is the standard library, and a shell that cannot import
  torch cannot quietly start depending on it.
- `nix build` runs the same gate as a derivation: it fails when a check fails, it cannot
  reach the network, and its source tree is read-only. Those are a CI job's constraints, met
  before there is a CI job. It is a check rather than an artefact - the only output is a
  timestamp - so `nix build --no-link` is the habit this repo documents: same semantics,
  without the `./result` symlink appearing in the working tree.
- `huggingface_hub` and PyYAML come from that same pin, so `python3 datasets/pull.py gsm8k`
  works in the shell with nothing installed by hand. The tool imports them inside its
  functions, which is why the gate can run offline with neither of them present. See
  `datasets/README.md`.
- `nix develop` and `nix build` only see files git knows about, so `git add` a new file
  before building it.

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
flake.nix        the pinned toolchain; nix build = the same gate as a derivation
common/          schema, mask, verifier protocol, fusion, run records, contamination, checks (+ 126 tests)
p1-verifiers/    src/ fixtures/ checks/ configs/ env/ tests/   (harness, live path, Chapter 7's health report)
p2-reagent/      src/ fixtures/ checks/ tests/   (the credit-assignment loop; the policy is a table)
p3..p6/          one README each, written with their chapter
datasets/        pull scripts + manifest + digests + rows_jsonl.py (JSONL views the stdlib checks can read)
scripts/         verify.sh (the gate), check_repo.py (characters, payloads, secrets, selftest)
```

## Licence

MIT for the code (`LICENSE`). Datasets keep their own licences - see `datasets/README.md`.
