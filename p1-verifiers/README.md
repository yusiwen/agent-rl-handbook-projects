# P1 - a verifier suite and its coverage report

**Status: harness complete, one verifier implemented, three are yours.**
Book chapter: Chapter 5 (reward engineering) and Chapter 7 (task sets).
Compute tier: ✅ - runs on the reference machine as written.

## What you produce

Two things the book asks for and this project grades:

1. **A verifier suite** that can score your task set: `src/p1_verifiers/verifiers.py`.
   `ExactMatchVerifier` is done, as the worked example. `NumericVerifier`,
   `UnitTestVerifier` and `LogicPuzzleVerifier` raise `NotImplementedError` with what to
   do; each also declines every task it is asked about, so the coverage report stays
   honest while they are unfinished.
2. **A coverage report** over your task set: how many tasks anything can score, and the
   reasons for the rest. `python3 checks/check_p1.py` prints it, and the number it prints
   is an upper bound on what training can add - a task nothing can score is a task no
   amount of RL will teach.

## Run it

```bash
python3 p1-verifiers/checks/check_p1.py          # the grader: fixtures, no network, no GPU
python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests
```

The fixture run is deliberately offline. If it fails, the problem is here rather than in
your environment; if it passes and a real run fails, the problem is in the environment or
in the model. That ordering is the whole reason the fixtures exist.

## What the run does

```
fixtures/tasks.jsonl      11 tasks: 10 exact-match, 1 numeric (not yet scoreable)
fixtures/responses.jsonl  2 attempts per exact-match task, from a fixture responder
      |
      v
  responder -> verifier -> reward terms -> rollout row -> run record
      |
      v
runs/<run-id>/{header.json, rounds.jsonl, round-000.jsonl}
```

The fixture is built, not found: 6 of its 10 groups contain two attempts that disagree
(60% mixed). A task set where every attempt scores the same produces a zero advantage in
every group, and a training run on it will look alive while learning nothing. The check
script asserts these numbers, so an accidental edit to the fixtures fails loudly rather
than quietly changing what the run demonstrates.

`runs/` is gitignored: the row *format* is versioned, the rows are not.

## The three verifiers you owe

| Verifier | Use it when | The trap |
|---|---|---|
| `numeric` | the answer is a number and a tolerance is acceptable (`1024` vs `1024.0`) | parsing "the last number in the text" silently accepts `150 minutes` for `150` |
| `unit_test` | the task ships tests that return 0 on success | a hung sandbox must count as a failure, not as an error; run it in a container with a timeout |
| `logic_puzzle` | the answer is a structured assignment (a grid, a schedule) | a near miss and a wrong answer must not both collapse to zero, or you cannot tell them apart later |

`src/p1_verifiers/verifiers.py` explains each in the docstring of its stub.

## Configs

`configs/spark-0.6b.yaml` is the config that must run on the reference machine (a DGX
Spark). `configs/rented-8b.yaml` is for a rented GPU and is marked `read, do not run
here`. The difference between the two is the compute tier from the book's appendix.

## What is not here yet

- The real task set: `fixtures/` is 11 tasks, enough to prove the pipeline. Pull the real
  datasets with `../datasets/` (see its README) once you have measured your own
  throughput - the budget in `configs/` has a field waiting for that number.
- A model. The fixture responder stands in for one so the checks stay offline; a real run
  points the harness at a served model instead.
