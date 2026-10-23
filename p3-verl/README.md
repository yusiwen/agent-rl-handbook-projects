# P3 - the training runtime, one run deep

**Status: the loop is closed - served sampling, in-process update, HTTP sync - measured; the
framework run is not started.** Written
with Chapter 9.
Book chapter: Chapter 9 (industrial training runtime), project 3 in the book's appendix.
Compute tier: ⚠️ - single-GPU, small-model runs only. Multi-node and Megatron paths are
reading material, not exercises (see the book's compute appendix).

## What you build

One GRPO run, then one DPO run, through a real framework runtime - with the *memory
ledger* the book's fourth milestone asks for:

| Model in the run | Weights | Gradients | Optimiser | Activation peak | Total |
|---|---|---|---|---|---|
| policy | | | | | |
| reference | | | | | |
| reward / critic (if any) | | | | | |
| rollouts' KV cache | | | | | |

Fill the table with arithmetic first, then correct it with `nvidia-smi`/`torch.cuda` peaks
during the run. The gap between the two columns is the lesson.

## The part Project 2 leaves out, built

Project 2 has two halves that never meet: a served loop that reads weights over HTTP and never
updates them, and an updating loop that trains where it generates. The join is weight sync, and it
is the smallest interesting piece of a training runtime. It is built here, before the framework,
because the framework will hide it.

Two honest mechanisms, measured on the reference Spark on 2026-10-20 with
`checks/sync_cost.py` (JSON: `fixtures/sync-cost-20261020.json`):

| mechanism | cost | what else it costs |
| --- | --- | --- |
| restart the server with merged weights | **156 s** (168.5, 147.4, 156.4 s; median) | the KV cache, the CUDA graphs, every warm connection |
| load an adapter into the resident base | **16-18 ms** (25.7 MB, two loads) | nothing; the base weights never move |

The ratio is about **9,000x**, and it is the reason a LoRA-based RL loop can afford to sync every
round while a full-weight loop cannot. The design that follows:

```
round N:  trainer continues from adapter-rN-1, writes adapter-rN
          POST /v1/load_lora_adapter {"lora_name": "policy-rN", "lora_path": "/models/rl-c/adapter-rN"}
round N+1: every request names "policy-rN"
```

The adapter is **cumulative** - `save_pretrained` writes the adapter's current weights, not a delta -
so there is nothing to merge and no ordering to get wrong. The paths differ per container: the
trainer mounts the volume at `/work`, the server at `/models`.

Three things that cost a debugging cycle each, all written into `env/serve_lora.sh`:

1. `--enable-lora` is not enough. The runtime routes are gated on `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`;
   without it the server loads LoRA kernels and answers **404** on `/v1/load_lora_adapter`.
2. `--max-lora-rank` must be at least the adapter's rank (16 here), or the load fails with a shape
   error rather than a suggestion.
3. The adapter path is resolved **inside** the container, so `/work/rl-c/adapter-r6` and
   `/models/rl-c/adapter-r6` are the same bytes under two names.

**How you know it applied.** Not the HTTP 200, and not the generated text: on three real task
prompts, the base model and the SFT adapter produced the *same first action*, while their top-5
token distributions differed by more than a nat. `checks/sync_cost.py` compares those distributions
and reports the maximum delta (4.72 and 5.05 nats for the two adapters). A one-sample comparison
would have called a working adapter broken; trusting the status code would have called a broken one
working.

## verl, prepared rather than installed

Chapter 9 reads one runtime end to end, and the project's other half is that runtime. Preparation
here means answering "can this machine run it at all" before downloading anything large.

| question | answer |
| --- | --- |
| which version, pinned | **0.9.1**, tag `v0.9.1` = commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` |
| does it resolve on aarch64 / python 3.12 | **yes** - a dry-run resolution in the training image found wheels for every requirement (`fixtures/verl-0.9.1-resolution.txt`) |
| what it costs | ~140 packages on top of torch, including `ray[default]` 2.59, `transformers` 5.10.4, `numpy` 2.5, `pyarrow` 26 |
| what it does **not** declare | `torch` (the environment's, which here is an NGC build) and `vllm` (so the default vLLM rollout backend is absent) |

Two consequences, both recorded before any install:

1. **transformers 5.10.4, not 5.20.1.** verl pins `<5.11`, and the P2 environment carries 5.20.1,
   so verl needs its own image or venv. Two projects, two environments - not a merge.
2. **the first run should not use the vLLM rollout backend.** It is not installed, and adding it
   means matching vLLM to an NGC torch on aarch64. The HF rollout backend
   (`rollout.name=hf`, `attn_implementation=sdpa`) needs neither vLLM nor flash-attn, and the
   point of the first run is the training path, not the sampler.

What is still unknown, and stays unknown until something runs: whether an NGC torch 2.14.0a0 build
satisfies verl's expectations (no pin says), and what the memory ledger actually costs on one GPU.

## The loop, closed

`env/rl_served.py` is the join: each round samples over HTTP from whatever adapter the server is
serving, computes the gradient here, and registers the result as the next round's adapter. Measured on
the Spark on 2026-10-20 (record `runs/p3-rl-served-2`, `code_sha git:50b37ab`), six calibrated tasks,
five samples each, three epochs per round, learning rate 1e-4, starting adapter `policy-sft`:

| round | served | solved | spread | loss | weighted turns | calls | sync | round time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `policy-sft` | 6/30 | 0.83 | -1.39028 | 90 | 120 | 12 ms -> `policy-r1` | 45 s |
| 1 | `policy-r1` | **25/30** | 0.00 | 0.0 | 0 | 70 | 13 ms -> `policy-r2` | 12 s |
| 2 | `policy-r2` | 25/30 | 0.00 | 0.0 | 0 | 70 | 13 ms -> `policy-r3` | 12 s |

It was run twice, and the second run is the one cited because it is the one with a clean record and a
correct project attribution. The first (`code_sha git:e056f49`) started from **11/30** and landed in
the same place:

| run | round 0 | round 1 | round 2 |
| --- | --- | --- | --- |
| first | 11/30 | **25/30** | 25/30 |
| second | 6/30 | **25/30** | 25/30 |

Two starts twenty points apart, one identical endpoint. **That plateau was the budget, not a wall.**
The same loop at ten samples per task (record `runs/p3-rl-served-5`, `code_sha git:b0bd83a`):

| round | served | solved | files-r05 | r06 | r07 | r09 | ledger-09 | ledger-3s02 | peak | round time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `policy-sft` | 18/60 | 3/10 | 4/10 | 6/10 | **0/10** | 4/10 | 1/10 | 8.9 GiB | 71 s |
| 1 | `policy-r1` | **60/60** | 10/10 | 10/10 | 10/10 | **10/10** | 10/10 | 10/10 | 8.9 GiB | 25 s |
| 2 | `policy-r2` | 60/60 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 8.9 GiB | 26 s |

The task that was 0 of 10 in round 0 - the one the five-sample budget could never train on - was
10 of 10 after one update. At five samples its group came back all-wrong, the advantage was exactly
zero, and it contributed no gradient at all; at ten samples three other tasks produced partial groups
(3 of 10, 1 of 10), the update generalised, and the wall fell with them. **A task that a group cannot
see is not a wall; it is an unmeasured task.** The ceiling that remains at 60/60 is the task set, not
the policy, and telling those apart needs harder tasks, not more rounds.

The server ends the run holding `qwen3-1.7b-base`, `policy-sft`, `policy-r1`, `policy-r2`,
`policy-r3`, and the per-round line is the lineage: the name printed as `served=` in round N+1 is the
name registered at the end of round N.

Three things this establishes, and one it exposes:

- **The sync is not the bottleneck.** 12-13 ms against a 156 s restart, inside a 12 s round in the
  flat rounds - the sampling and the gradient are what take the time.
- **Feedback reaches the next round's samples.** 6/30 (and, in the other run, 11/30) to 25/30 after
  one update, and nothing stopped the loop: the next round ran on the new adapter without a restart or
  a reconnection.
- **The loop stops itself when there is nothing left.** Five of six groups had a non-zero spread in
  round 0 (0.83); afterwards every group was flat at 5/5 or 0/5, the advantage was exactly zero, and
  the last two rounds trained on nothing - the ceiling arriving, as in Project 2, with a diagnostic
  number (`spread_fraction`) instead of a mystery.
- **The budget is still five samples per task.** 25/30 is not "solved": the five remaining failures
  are a task at 0/5. Ten samples answered that question - the task went to 10/10 - and the next rung,
  five-step plans, is where the loop genuinely cannot start (below).

## Where the ceiling actually is

Two rungs were tried above the calibrated band. The first was not a rung at all: the six tasks the SFT
policy fails 60 of 60 times at `temperature 1.0` (`runs/p3-rl-served-6`, `code_sha git:2042a8c`).
Every group is a constant, so the advantage is exactly zero and the update is exactly zero - a wall in
the algorithm's precondition, not in the task. At `temperature 1.7` the same six produce 4 of 60
(`-8`), one update takes them to 60 of 60 (`-9`), and the learned adapter solves all sixty greedily as
well (`-10`, `reward_mean` 1.0): it is a policy, not a lucky sampler. That also explains the earlier
five-sample plateau from the other side - those groups were all-wrong at five samples too, and an
all-wrong group contributes no gradient.

The second rung is real. `fixtures/hard.jsonl` adds three five-step ledger plans - five moves plus a
report, exactly the six-turn budget - and three files tasks over five-file directories
(`code_sha git:1a4a822`):

| set | greedy (T=0) | T=1.7 |
| --- | --- | --- |
| files, five-file directories | **30/30** | 24/30 |
| ledger, five-step ordered plan | **0/30** | 0/30 |

The ledger rung fails in a specific and instructive way: the reported totals are right (5 and 8 in ten
of ten samples at T=0; 8 in ten of ten at T=1.7) and `goal_met` is false in every one, because the plan
is applied out of order. The model sums the five movements and announces the sum; it never executes
the sequence, and it never fails to add up. **Zero successes in 60 rollouts** means no reward variance
at either temperature, therefore no advantage, therefore no gradient - the wall the book describes when
it says a group cannot see a task: not "too hard", but "no sample succeeds, so there is nothing to
compare".

And it is a data wall, not a capacity wall: the arithmetic is right and the behaviour is wrong. Two
things could break it, neither tried here - demonstrations at five steps, which is what SFT is for, or
a process reward on the ordered prefix, which has variance even when the outcome never does. Until one
of them is run, the honest statement is that outcome-only GRPO at this budget cannot start on a
five-step ordered plan, and the ceiling it measures is the reward's, not the model's.

## The server gets a ceiling too

`SERVER_MEM` (default 32 GB) caps the inference container, for the same reason the training container
is capped: the pool is unified, and a server without a limit is not a sandbox but a coinhabitant. 1.7B
in bf16 is 3.4 GB, the KV reservation at `--gpu-memory-utilization 0.10` is about 13 GB, and CUDA
context plus graphs add a few more - so 32 GB is generous and still bounded.

## Order matters

1. `transformers` + `peft` + `trl` first: these have aarch64 wheels, so a failure is your
   code.
2. Then the framework the chapter uses (VeRL), where attention must be set to `sdpa` and a
   `flash_attention_2` default will fail for reasons that have nothing to do with RL.
3. DPO before GRPO: fewer moving parts, and it exercises the same data path.

`../p1-verifiers/` supplies the task set and the verifier suite; this project supplies the
runtime around them. The row format is unchanged - which is the point of `../common/`.

## What is not here yet

- **`run_p2_served.py` itself is still read-only.** The closed loop lives here in `p3-verl`, not in
  the Project 2 check; porting the per-round registration back is a one-argument change, and it is
  deliberately deferred so the same work is described in one place rather than two.
- **The framework run**: the two configs (`configs/spark-0.6b-grpo.yaml`, `configs/spark-0.6b-dpo.yaml`),
  the ledger filled from a real run, and a decision about which framework version to pin - that
  belongs with the first real run, not before it.

## Files

- `env/serve_lora.sh` - the server: pinned image, LoRA enabled, runtime updating allowed, the volume
  mounted read-write at `/models`.
- `checks/sync_cost.py` - restart cost, adapter cost, the distribution check, unload; `--json-out`
  for evidence.
- `fixtures/sync-cost-20261020.json` - the run behind the table above.
