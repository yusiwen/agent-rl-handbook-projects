# P1 - a verifier suite and its coverage report

**Status: the offline harness and the live path are both complete; three verifiers implemented,
one is yours.**
Book chapter: Chapter 5 (reward engineering) and Chapter 7 (task sets).
Compute tier: ✅ - runs on the reference machine as written.

## What you produce

Two things the book asks for and this project grades:

1. **A verifier suite** that can score your task set: `src/p1_verifiers/verifiers.py`.
   `ExactMatchVerifier`, `NumericVerifier` and `UnitTestVerifier` are done, as the worked
   examples. `LogicPuzzleVerifier` raises `NotImplementedError` with what to do; it also
   declines every task it is asked about, so the coverage report stays honest while it is
   unfinished.
2. **A coverage report** over your task set: how many tasks anything can score, and the
   reasons for the rest. `python3 checks/check_p1.py` prints it, and the number it prints
   is an upper bound on what training can add - a task nothing can score is a task no
   amount of RL will teach.

## Run it

```bash
python3 p1-verifiers/checks/check_p1.py          # the grader: fixtures, no network, no GPU
python3 -m unittest discover -s p1-verifiers/tests -t p1-verifiers/tests

# Against a served model (any OpenAI-compatible endpoint, e.g. vLLM on the reference machine):
python3 p1-verifiers/checks/run_p1_live.py \
    --base-url http://192.168.2.40:8000 --model qwen3-1.7b --no-thinking --samples 2
```

The fixture run is deliberately offline. If it fails, the problem is here rather than in
your environment; if it passes and a real run fails, the problem is in the environment or
in the model. That ordering is the whole reason the fixtures exist.

The live runner does five things worth knowing about before you point it at a GPU: it
**preflights the model name** against `/v1/models` (so a typo costs a second, not a round),
it **prints coverage before spending tokens**, it counts tokens with the **server's own
tokeniser** over `/tokenize` and puts that tokeniser's name in every row, it **runs the five
checks on the rows it just produced**, and it writes a run record naming the endpoint, the
model, the task-file digest and the commit. Exit status is non-zero if any check fails - the
record is still written, because a record of a rejected run is evidence.

## What the run does

```
fixtures/tasks.jsonl      12 tasks: 10 exact-match, 1 numeric, 1 unit-test
fixtures/responses.jsonl  2 attempts per task, from a fixture responder
      |
      v
  responder -> verifier -> reward terms -> rollout row -> run record
      |
      v
runs/<run-id>/{header.json, rounds.jsonl, round-000.jsonl}
```

The fixture is built, not found: 8 of its 12 groups contain two attempts that disagree
(67% mixed). A task set where every attempt scores the same produces a zero advantage in
every group, and a training run on it will look alive while learning nothing. The check
script asserts these numbers, so an accidental edit to the fixtures fails loudly rather
than quietly changing what the run demonstrates.

`runs/` is gitignored: the row *format* is versioned, the rows are not.

## The verifiers

| Verifier | Status | Use it when | The trap |
|---|---|---|---|
| `exact_match` | implemented | the answer has one agreed surface form (a number, a short name, a date) | the same value in a different surface form fails: `1024.0` is not `1024` |
| `numeric` | implemented | the answer is a number and a tolerance is acceptable (`1024` vs `1024.0`) | units are ignored, so `150 kilometres` scores the same as `150 minutes`; a task where the unit carries meaning wants `exact_match` |
| `unit_test` | implemented | the task ships tests that return 0 when the attempt is correct | the tests run in a sandbox, so the machine running the *runner* needs a container runtime; without one the run stops with an error instead of scoring a zero, because "this machine cannot run the tests" is not the model's fault |
| `logic_puzzle` | your turn | the answer is a structured assignment (a grid, a schedule) | a near miss and a wrong answer must not both collapse to zero, or you cannot tell them apart later |

`src/p1_verifiers/verifiers.py` explains each in its docstring - including what it will not
do, which is the part that decides whether you can trust the numbers it produces.

The unit-test verifier is worth reading in full before you point it at a model's output. The
attempt is written into a fresh temporary directory and the task's tests run there in a
container with `--network=none` and a memory cap, under a wall-clock timeout from outside; a
timeout is recorded as a *failure* whose detail says so, because an environment that hangs
must not look like a model that is slow. Every detail also names the sandbox that produced
it, so a row scored without a container is distinguishable from one scored inside one. The
registry hands out the container; the offline check passes a `LocalSandbox` instead, because
the fixture's attempts are written in this repository - and that sandbox refuses to be
anonymous: it demands a sentence saying whose code this is, and puts it in every detail it
produces.

## Configs and the environment

`configs/spark-0.6b.yaml` is the config that must run on the reference machine (a DGX Spark).
`configs/rented-8b.yaml` is for a rented GPU and is marked `read, do not run here`. The
difference between the two is the compute tier from the book's appendix.

`env/` is the server those configs describe: a pinned `docker-compose.yaml` that starts one
vLLM model, plus the four traps that cost a restart on this machine. Read `env/README.md`
before the first start - in particular, `nvidia/Qwen3-8B-NVFP4` is a base checkpoint, and the
compose's defaults are the ones this project was measured against.

## What is not here yet

- The real task set: `fixtures/` is 12 tasks, enough to prove the pipeline. Pull the real
  datasets with `../datasets/` (see its README) once you have measured your own
  throughput - the budget in `configs/` has a field waiting for that number.
- **Someone to start the server.** `env/docker-compose.yaml` is ready and pinned, and the code is
  wired to it, but starting it is your call - and the machine has to be on. Two commands:
  `docker compose up -d` in `env/`, then the runner, which preflights the model name against
  `/v1/models` and the tokeniser against `/tokenize` before it spends anything. The wiring has
  been proven end to end (2026-10-10); what a live model then scores is the model's number, not
  this fixture's.
- The last verifier (`logic_puzzle`), which is the one that needs a parser rather than a
  comparison, and which is the only name left in `UNIMPLEMENTED`.
- **A container runtime where the runner runs, for `unit_test`.** This machine deliberately runs
  no docker, so a live run started here would raise `SandboxUnavailable` the moment it reached a
  unit-test task, rather than running generated code directly on the host. That behaviour is the
  point, and it is also a step to plan for: the machine that judges the tests is the one that
  needs the sandbox.
- A second sampling round. One round gives you rows and an advantage spread; the round-stability
  check needs two rounds with the same sample ids, which is what `common/runs.py` records.
- A model that produces a spread at this size. One live round on the reference machine
  (1.7B, two samples per task) left every group flat - same reward both times - which is a
  fact about the task set and the sampler, not about the verifier: `--samples 4` and a
  harder set are the levers, and `check_advantage_spread` is the check that says so.
