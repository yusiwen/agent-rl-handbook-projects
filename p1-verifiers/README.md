# P1 - a verifier suite and its coverage report

**Status: complete. The offline harness, the live path, and all four verifiers are in place.**
Book chapter: Chapter 5 (reward engineering) and Chapter 7 (task sets).
Compute tier: ✅ - runs on the reference machine as written.

## What you produce

Two things the book asks for and this project grades:

1. **A verifier suite** that can score your task set: `src/p1_verifiers/verifiers.py`. All four
   are implemented - `ExactMatchVerifier`, `NumericVerifier`, `UnitTestVerifier` and
   `LogicPuzzleVerifier` - and each docstring says what its verifier will *not* do, which is
   the part that decides whether you can trust the numbers it produces.
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
fixtures/tasks.jsonl      18 tasks: 12 numeric, 2 exact-match, 2 unit-test, 2 logic-puzzle
fixtures/responses.jsonl  2 attempts per task, from a fixture responder
      |
      v
  responder -> verifier -> reward terms -> rollout row -> run record
      |
      v
runs/<run-id>/{header.json, rounds.jsonl, round-000.jsonl}
```

The fixture is built, not found: 16 of its 18 groups contain two attempts that disagree
(89% mixed), and the two that do not are kept deliberately - one task this responder is
reliable on, one it cannot do at all - because a fixture whose every group moves would be a
claim about a model rather than a demonstration of the machinery. A task set where every attempt scores the same produces a zero advantage in
every group, and a training run on it will look alive while learning nothing. The check
script asserts these numbers, so an accidental edit to the fixtures fails loudly rather
than quietly changing what the run demonstrates.

`runs/` is gitignored: the row *format* is versioned, the rows are not.

### Why the task set looks like this

The set was changed twice, and both times what changed it was a live run against a 1.7B model
rather than an opinion about difficulty. The numbers are in the last section.

**The arithmetic answers are judged as numbers.** Every arithmetic group came back flat, and the
arithmetic was mostly *right* - the text was not: prose around the number, a unit after it,
`1024.0` where the fixture said `1024`. Comparing a numeric answer by text measures formatting,
so those tasks declare `numeric` with a tolerance of `0`.

**Then the reward had to stop depending on where a sentence ends.** The first run under that rule
produced two verdicts that could not be defended, both from real output: `"There are 150 minutes
in 2.5 hours."` is correct and scored `0.0` on every sample, because the last number in it is
`2.5`; and four identical tank answers split two passes and two failures, because two of them
ended at the litres and two ended at the minutes. So `numeric` now requires the answer to be
*declared* - the final non-empty line, carrying exactly one number - records that fact in every
detail as `form=ok` or `form=no`, and never scores a value it could not read. What that costs is
in the verifier table above, and this file does not pretend otherwise: a correct answer written
as prose is a zero, and the zero says why.

**Five tasks were added at this model's edge.** Two of them sit there: the sum-of-a-series
question, which flips between `210` and `155`, and the logic puzzle. The other three are
instructive rather than useful - two are too easy for this model (flat at 1.0 every time) and
the letter count is beyond it (a deterministic `4`, flat at 0.0).

## The verifiers

| Verifier | Use it when | The trap |
|---|---|---|
| `exact_match` | the answer has one agreed surface form (a number, a short name, a date) | the same value in a different surface form fails: `1024.0` is not `1024` |
| `numeric` | the answer is a number, the task asks for it on its own last line, and a tolerance is acceptable (`1024` vs `1024.0`) | the answer must be **declared**: the final line has to carry exactly one number, so `"...holds 36 litres after 12 minutes"` scores zero even though 36 is in it. Every detail says which half passed (`form=ok` / `form=no`) and names the numbers it saw, so a flat group can be attributed instead of guessed at. Units are ignored, so `150 kilometres` scores the same as `150 minutes` |
| `unit_test` | the task ships tests that return 0 when the attempt is correct | the tests run in a sandbox, so the machine running the *runner* needs a container runtime; without one the run stops with an error instead of scoring a zero, because "this machine cannot run the tests" is not the model's fault |
| `logic_puzzle` | the answer is a structured assignment and the task can state constraints over it | the constraints *are* the answer: a near miss scores a hard zero, so the detail is where the gradient is - it names the constraint that failed and why - and a solution with no variables, no constraints or an unknown kind is an error rather than a task that scores zero |

`src/p1_verifiers/verifiers.py` explains each in its docstring.

Two of them are worth reading in full before you point them at a model's output.

The **unit-test** verifier writes the attempt into a fresh temporary directory and runs the
task's tests there in a container with `--network=none` and a memory cap, under a wall-clock
timeout from outside; a timeout is recorded as a *failure* whose detail says so, because an
environment that hangs must not look like a model that is slow. Every detail also names the
sandbox that produced it, so a row scored without a container is distinguishable from one
scored inside one. The registry hands out the container; the offline check passes a
`LocalSandbox` instead, because the fixture's attempts are written in this repository - and
that sandbox refuses to be anonymous: it demands a sentence saying whose code this is, and
puts it in every detail it produces.

The **logic-puzzle** verifier parses an assignment out of the response - the last JSON
object, or `name = value` lines, keeping only the variables the task declares - and checks it
against the task's constraints one kind at a time: `distinct`, `equals`, `not_equals`. A
variable the answer never mentions fails whatever constraint names it, because an incomplete
assignment is not vacuously right about the parts it left out. It scores a hard zero for
anything short of all constraints holding, for the same reason `numeric` refuses partial
credit for wrong arithmetic: a reward you cannot explain later is worse than one you can
defend. The fraction that held is not lost, though - it is in the detail, so a near miss
reads like this in a run record:

```
satisfied 2 of 3 constraints; failed: c1 (distinct: alice and bob are both 'blue')
```

## Configs and the environment

`configs/spark-0.6b.yaml` is the config that must run on the reference machine (a DGX Spark).
`configs/rented-8b.yaml` is for a rented GPU and is marked `read, do not run here`. The
difference between the two is the compute tier from the book's appendix.

`env/` is the server those configs describe: a pinned `docker-compose.yaml` that starts one
vLLM model, plus the four traps that cost a restart on this machine. Read `env/README.md`
before the first start - in particular, `nvidia/Qwen3-8B-NVFP4` is a base checkpoint, and the
compose's defaults are the ones this project was measured against.

## What is not here yet

- The real task set: `fixtures/` is 18 tasks, enough to prove the pipeline. Pull the real
  datasets with `../datasets/` (see its README) once you have measured your own
  throughput - the budget in `configs/` has a field waiting for that number.
- **Someone to start the server.** `env/docker-compose.yaml` is ready and pinned, and the code is
  wired to it, but starting it is your call - and the machine has to be on. Two commands:
  `docker compose up -d` in `env/`, then the runner, which preflights the model name against
  `/v1/models` and the tokeniser against `/tokenize` before it spends anything. The wiring has
  been proven end to end (2026-10-10); what a live model then scores is the model's number, not
  this fixture's.
- **A container runtime where the runner runs, for `unit_test`.** This machine deliberately runs
  no docker, so a live run started here raises `SandboxUnavailable` the moment it reaches a
  unit-test task rather than running generated code on the host. That behaviour is the point, and
  it is also a step to plan for: the machine that judges the tests is the one that needs the
  sandbox. Verified on a machine that has one - the reference DGX Spark, 2026-10-13: all 18 tasks
  scored, both unit-test tasks 20 of 20 attempts across three sampler configurations, every row
  reading `tests passed (docker; attempt written to main.py)`. That run earned its keep by
  breaking in a way no fixture sandbox can: docker runs as root, the model's attempt wrote
  `__pycache__` into the bind mount *as root*, and the verifier's cleanup then raised
  `PermissionError` and killed the run instead of reporting a score. The container now runs as
  the caller (`--user`), and cleanup is housekeeping that cannot fail a score - both in the
  `DockerSandbox` docstring, and both tested.
- A second sampling round. One round gives you rows and an advantage spread; the round-stability
  check needs two rounds with the same sample ids, which is what `common/runs.py` records.
- **A model that produces a spread at this size, and a reward that can be believed.** Measured
  on the reference machine, thinking off, over 16 scoreable tasks - the two `unit_test` ones need
  a container runtime, which this machine deliberately does not have.
  *Round one*, with the loose rule and no format instruction in the prompts: 2 samples at
  temperature 0.7 left 0 of 16 groups mixed, 4 samples at 0.7 left 1, 4 samples at 1.2 left 1.
  Reading the rows rather than the summary is what showed the wrong thing was being fixed: the
  one "mixed" group had split because two sentences ended at the value and two ended at the
  minutes, and a *correct* task was flat at 0 because the last number in a one-line answer was
  `2.5` instead of `150`.
  *Round two*, with the format gate and the tasks asking for the form: 1, 2 and 4 mixed groups
  out of 16 - and the pass rate *fell* (78%, 75%, 72% against 81%, 79%, 84%), because the reward
  stopped paying for answers it never read. Two of the four movers are still shape rather than
  skill (one sample declares its answer, another closes with `$$`); the ones that are not are the
  two worth training on: the sum-of-a-series question that flips between `210` and `155`, and the
  four-seat puzzle that flips between 3 of 4 constraints holding and 4 of 4.
  *A third run, on the reference machine, covered all 18 tasks with `unit_test` included*: 1, 3 and
  5 mixed groups out of 18; pass rates 80%, 79% and 75%; both unit-test groups flat at 1.0, because
  this model writes `add` and `fizzbuzz` correctly every time. It also corrected a claim from the
  16-task run: the letter count is **not** beyond the model - it answered `4` four times on this
  Mac and `3`, `5`, `4`, `5` on the Spark, so it is an edge task and its group moves. After both
  runs the movers are the series sum, the four-seat puzzle, the letter count, and one unit-price
  word problem whose movement is form rather than skill.
  The same run measured a second lever, and it is the cheapest one so far: **asking for the format
  in the task works at this model size.** With one sentence added to the prompt, the model that
  had answered `"There are 150 minutes in 2.5 hours."` started answering `150` - a task scoring
  0 of 4 because of the verifier became a task scoring 4 of 4 because of the prompt. 79-88% of
  numeric attempts now declare their answer. The other side of that is the honest cost: a correct
  sentence that declares nothing scores zero, and `check_advantage_spread` plus the flat-group
  attribution printed by `checks/check_p1.py` are what say so instead of leaving it to be
  discovered during a training run.
