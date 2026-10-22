# P3 - the training runtime, one run deep

**Status: the weight-sync half is built and measured; the framework run is not started.** Written
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

## Order matters

1. `transformers` + `peft` + `trl` first: these have aarch64 wheels, so a failure is your
   code.
2. Then the framework the chapter uses (VeRL), where attention must be set to `sdpa` and a
   `flash_attention_2` default will fail for reasons that have nothing to do with RL.
3. DPO before GRPO: fewer moving parts, and it exercises the same data path.

`../p1-verifiers/` supplies the task set and the verifier suite; this project supplies the
runtime around them. The row format is unchanged - which is the point of `../common/`.

## What is not here yet

- **The loop wired to the sync.** `run_p2_served.py` still registers no adapter; the next step is a
  per-round registration and a request that names the new adapter, which turns `Config.served` from
  read-only into updated.
- **The framework run**: the two configs (`configs/spark-0.6b-grpo.yaml`, `configs/spark-0.6b-dpo.yaml`),
  the ledger filled from a real run, and a decision about which framework version to pin - that
  belongs with the first real run, not before it.

## Files

- `env/serve_lora.sh` - the server: pinned image, LoRA enabled, runtime updating allowed, the volume
  mounted read-write at `/models`.
- `checks/sync_cost.py` - restart cost, adapter cost, the distribution check, unload; `--json-out`
  for evidence.
- `fixtures/sync-cost-20261020.json` - the run behind the table above.
