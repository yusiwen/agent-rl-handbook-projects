# The training environment for Project 2

Supervised fine-tuning of a 1.7B policy needs PyTorch built for this machine, and this machine is
awkward: aarch64, a Blackwell Ultra GPU (GB10, compute capability 12.1), 128 GB of unified memory.
The wheels on the PyPI index are not built for it. NVIDIA's NGC image is, so the environment is a
container, and the host is left alone: no virtualenv, no site-packages, no system Python touched.

Three files, each doing one thing:

| file | what it is for |
| --- | --- |
| `Dockerfile` | the base image pinned by tag and digest, plus the two packages the loop imports |
| `build.sh` | builds it and prints the image id, so "which image" is never a guess |
| `run.sh` | the `docker run` flags this host needs, and the three mounts that keep the host clean |
| `probe.py` | what the image can actually do; exit status 0 only if it all works |

## Measured on 2026-10-18, on `192.168.2.40`

`bash build.sh`, then `bash run.sh`, then
`bash run.sh python3 /env/probe.py --checkpoint Qwen/Qwen3-1.7B`.

| check | result |
| --- | --- |
| torch | `2.14.0a0+b2c75dd062.nv26.09`, built for CUDA 13.4 |
| python | 3.12.3 |
| device | NVIDIA GB10, compute 12.1, 121.6 GiB |
| compiled archs | `sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120` - **no `sm_121`** |
| float32 matmul, TF32 on (the default here) | relative error `2.92e-04` |
| float32 matmul, TF32 off | relative error `8.80e-07` |
| bfloat16 matmul, 4096-cubed | runs, `(4096, 4096)` |
| bfloat16 throughput, 2048-cubed x 50 | 91.6 - 96.5 TFLOP/s over three runs |
| scaled dot-product attention, causal | runs, finite output |
| backward pass + AdamW, 10 steps | loss `0.0386 -> 0.0021`, all gradients finite |
| LoRA (r=8, a 2-layer Qwen3 model), 10 steps | `7,168 / 433,920` trainable (1.65%), loss `6.25 -> 5.82`, adapter saved and reloaded |
| Qwen3-1.7B, one forward pass | load 27 - 32 s, forward 0.08 - 0.20 s, 1,720,574,976 parameters, peak 4.07 - 4.14 GiB |
| image size | base 25.3 GB, this image 25.4 GB |

Where a range is given, the spread is run-to-run on the same image, not a difference in setup;
the checks were re-run from the committed files, not from the transcript that first produced them.

## Three things this measured that are worth keeping

**The architecture list is not the answer to "will it run".** `archs` stops at `sm_120` while the
device reports compute 12.1, and a string comparison against `sm_121` says the GPU is unsupported.
It is not: the image carries PTX for `compute_120`, the driver JIT-compiles it for the real
architecture, and every check above passes - matmul, attention, backward, optimizer step, a LoRA
adapter, a 1.7B forward pass. The probe therefore tests kernels, not strings, and
`--require-sm121` is an opt-in flag for a machine whose driver genuinely cannot JIT.

**A "float32" matmul here is TF32.** The NGC images export `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, so
the default product carries about `1e-4` relative error. That is a property of the image, not of
the hardware: with TF32 disabled the same product is exact to `8.8e-07`. A check that assumes
`1e-4` without knowing this reports a healthy machine as broken, which is exactly what happened on
the first run of this probe.

**What the image already has, and what it does not.** It ships `triton 3.8.0`,
`flash-attn 2.8.3.post1`, `safetensors 0.8.0` and `datasets 4.7.0`, and it does not ship
`transformers`, `peft`, `trl` or `accelerate`. The build adds `transformers` and `peft` at pinned
versions and nothing else, because the training loop is written in this repository rather than
called out of `trl`: the thing Project 2 studies is how credit reaches the loss, and a trainer
class hides that behind one call.

## Running it

```bash
bash p2-reagent/env/build.sh                                  # idempotent; seconds on a warm cache
bash p2-reagent/env/run.sh                                    # the probe without a checkpoint
bash p2-reagent/env/run.sh python3 /env/probe.py --checkpoint Qwen/Qwen3-1.7B
bash p2-reagent/env/run.sh python3 /env/probe.py --require-sm121   # strict, for another host
```

Cover a different cache, volume or image with `ARLH_HF_CACHE`, `ARLH_VOLUME`, `ARLH_TRAIN_IMAGE`.

## Where the host is allowed to be touched, and where it is not

- the Hugging Face cache is mounted **read-only** at `/hf`, so a load cannot write to the host;
- the repository's `env/` directory is mounted **read-only** at `/env`, so the code that ran is the
  code that was committed;
- outputs go to the named Docker volume `arlh-p2` at `/work`, which is removable in one command and
  is not a directory in the host's file tree.

## P0b: ten LoRA steps on the real checkpoint

`bash run.sh python3 /env/smoke_lora.py`, 2026-10-18, same machine.

| quantity | value |
| --- | --- |
| trainable parameters | `6,422,528 / 1,726,997,504` (0.372%, r=16 on q/k/v/o) |
| loss, step 1 to step 10 | `13.7416 -> 0.0000` |
| load / train | 32.8 s / 1.9 s total, 0.19 s per step |
| throughput | 2,742 tokens/s (batch 2, 256 tokens, bf16) |
| peak GPU memory | 5.90 GiB allocated, 6.23 GiB reserved |
| adapter | saved to `/work/p0b-adapter`, reloaded as `PeftModelForCausalLM` |

The loss mask is checked rather than assumed: the smoke prints the first sample's prompt and its
supervised span, and they are

```
prompt  '<|im_start|>user\nWhat is 17 + 26?<|im_end|>\n<|im_start|>assistant\n'
labels  '43<|im_end|>'
```

### Why step one is 13.7, and why that is the point

A loss of 13.7 on the first step looks like a defect. `--inspect` says it is the data doing its
job, per token:

```
first supervised token: '4' at position 22
  model wants  '1' (logit 42.0), 'Let' (30.3), 'To' (28.3), '$' (28.1), 'The' (27.1)
  '4'          ce 18.375
  '3'          ce  0.000
  '<|im_end|>' ce  3.538
```

The checkpoint is post-trained (`base_model: Qwen/Qwen3-1.7B-Base`), and after
`<|im_start|>assistant\n` it still wants to restate or explain the question before answering. So the
first answer token carries almost none of the probability mass, and the cross entropy is large.
That is the same behaviour Chapter 7 measured on this checkpoint from the other direction - the
model ignoring a stated answer format - and it means the fine-tune has something real to teach:
the format, not the arithmetic.

## P2-prime: the demonstration fine-tune

`bash run.sh python3 /env/sft.py`, 2026-10-18, same image.

| quantity | value |
| --- | --- |
| demonstrations | 43 tasks, 117 turns, `sha256:41d694a18fd7ffa4` |
| supervised span | the reply only - prompt tokens are masked |
| trainable | `6,422,528 / 1,726,997,504` (0.372%, r=16 on q/k/v/o) |
| loss | `3.8492 -> 0.0000` over 59 steps |
| time / memory | 23.7 s, 9.50 GiB peak |
| artifacts | `/work/sft/adapter` and `/work/sft/merged` |

The merged checkpoint is served by the same vLLM image with `--gpu-memory-utilization 0.08`, on
port 8001 beside the baseline on 8000: 3.38 GiB each, both resident at once.

## Step A: does a gradient reach the next round?

`bash run.sh python3 /env/rl_step.py --task files-r06 --samples 5 --rounds 2`, 2026-10-18, starting
from the P2-prime adapter, `lr 1e-5`, in-process generation.

| round | solved | mean reward | loss | weighted turns | adapter digest |
| --- | --- | --- | --- | --- | --- |
| 0 | 4/5 | +0.800 | -0.15491 | 15 | `sha256:5a15df58...` |
| 1 | 5/5 | +1.000 | 0.00000 | **0** | unchanged |

Two things it proves and one it exposes.

**Proves:** the weight the credit machinery computes reaches the tokens the model actually produced
- the adapter digest moves from `sha256:4d51af56...` to `sha256:5a15df58...` - and the next round
generates from those weights, because generation and training share one process. There is no
weight-sync step to get wrong; it is structural, not verified after the fact.

**Exposes:** round 1 produced no gradient at all. The group came back five-for-five, a flat group is
a zero advantage by design, so `weighted turns` is zero and the loss is exactly 0.0. One task with
five samples spends all of its variance in the first round: step B has to run on the band, not on a
single task taken from it.

**A sign worth not misreading:** the loss is negative in round 0. The objective is
`- w * mean(log p)`, so a negative advantage multiplied by a negative log-probability contributes a
negative number, and a round whose weighted sum is dominated by negative advantages has a negative
scalar. What trains is the gradient's sign, not the scalar's.

## Step B: eight rounds with the update switched on

`bash run.sh python3 /env/rl_step.py --tasks files-r05 files-r06 files-r07 files-r09 ledger-09
--samples 5 --rounds 8 --lr 1e-5`, 2026-10-18, from the P2-prime adapter, one step per round.

| round | solved | r05 | r06 | r07 | r09 | ledger-09 | loss | weighted turns | digest changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19/25 | 5 | 4 | 5 | 5 | 0 | -0.09921 | 14 | yes |
| 1 | 17/25 | 3 | 5 | 5 | 4 | 0 | -0.05868 | 25 | yes |
| 2 | 20/25 | 5 | 5 | 5 | 5 | 0 | 0.0 | 0 | no |
| 3 | 18/25 | 3 | 5 | 5 | 5 | 0 | -0.31342 | 10 | yes |
| 4 | 16/25 | 2 | 5 | 5 | 4 | 0 | -0.13500 | 23 | yes |
| 5 | 19/25 | 5 | 5 | 5 | 4 | 0 | 0.0 | 10 | yes |
| 6 | 19/25 | 5 | 5 | 5 | 4 | 0 | 0.0 | 10 | yes |
| 7 | 20/25 | 5 | 5 | 5 | 5 | 0 | 0.0 | 0 | no |

**No trend.** The solved share moved between 16 and 20 of 25 - eight points, which is the sampling
noise of twenty-five rollouts at temperature 1.0 - and the two rounds with zero weighted turns had
no gradient because every sample in every group succeeded: the flat ceiling, arriving in the training
loop exactly as it did in the evaluation loop.

Three things are established, and they are worth separating from the non-result:

1. **The update path works.** Six of eight rounds changed the adapter; the two that did not are the
   ones where the credit signal was legitimately zero.
2. **A wall is a wall.** `ledger-09` solved zero of forty attempts. Its calibration said 0.25, which
   is what a rate measured on twelve samples buys you: a task at the edge of the band behaves like a
   wall at ten.
3. **This budget cannot detect a policy change.** One step per round at `1e-5` over 0.372% of the
   parameters is eight small steps in total, with trajectory-level advantages giving every turn of an
   attempt the same sign. That is a design with no power to show improvement, and reporting "no
   improvement" from it would be as dishonest as reporting a win.

The next experiment needs a budget chosen for the question - more samples per group, more than one
step per round, a rate that can move an adapter, and no walls pretending to be band tasks - and it is
a separate run with its own record.

## Step C: a budget chosen for the question

`bash run.sh python3 /env/rl_step.py --tasks files-r05 files-r06 files-r07 files-r09 ledger-09
ledger-3s02 --samples 10 --epochs 3 --rounds 6 --lr 1e-4`, 2026-10-18.

| round | solved | r05 | r06 | r07 | r09 | ledger-09 | 3s02 | loss | weighted turns | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 42/60 (70%) | 7 | 10 | 9 | 10 | **1** | 5 | -0.68295 | 129 | 3 |
| 1 | **60/60** | 10 | 10 | 10 | 10 | **10** | 10 | 0.0 | 0 | 0 |
| 2-5 | 60/60 | 10 | 10 | 10 | 10 | 10 | 10 | 0.0 | 0 | 0 |

**One round of three steps at 1e-4 moved the solved share from 70% to 100%**, including the task
that had solved once in ten attempts in round 0 and zero times in forty the day before. The thing
called a wall was a wall for *that* policy, not for the updated one. Rounds 1-5 produced no gradient
at all, because a group at 100% has a zero advantage: the loop had finished and had nothing left to
learn on these six tasks.

Two readings are possible and the record supports only the first:

* **Supported:** on-policy advantage-weighted updates, with a budget that can move an adapter, fixed
  the six tasks they were measured on, in one round.
* **Not supported:** that the policy generalises to tasks it did not see. Six deterministic tasks
  measured before and after cannot show that. A held-out set is the next measurement, and until it
  runs the claim stays "the six tasks".

## Not proven yet

P0b proves a ten-step run fits, converges and leaves a loadable adapter. It does not prove the
P2 fine-tune: the task set, the demonstration volume, whether format compliance reaches 90% in one
epoch, and whether the resulting policy is any better at the tasks. Those numbers belong to P1-P3
and will be added here only when they exist.
