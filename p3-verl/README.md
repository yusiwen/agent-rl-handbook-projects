# P3 - the training runtime, one run deep

**Status: not started.** Written with Chapter 9.
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

## Order matters

1. `transformers` + `peft` + `trl` first: these have aarch64 wheels, so a failure is your
   code.
2. Then the framework the chapter uses (VeRL), where attention must be set to `sdpa` and a
   `flash_attention_2` default will fail for reasons that have nothing to do with RL.
3. DPO before GRPO: fewer moving parts, and it exercises the same data path.

`../p1-verifiers/` supplies the task set and the verifier suite; this project supplies the
runtime around them. The row format is unchanged - which is the point of `../common/`.

## What is not here yet

- The two configs (`configs/spark-0.6b-grpo.yaml`, `configs/spark-0.6b-dpo.yaml`) and the
  ledger template as a file to fill in.
- A decision about which framework version to pin; that belongs with the first real run,
  not before it.
