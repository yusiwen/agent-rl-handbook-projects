#!/usr/bin/env bash
# Run a command in the Project 2 training container, with the flags this host needs and with
# nothing written to the host.
#
#   bash p2-reagent/env/run.sh                                    # the probe, about a minute
#   bash p2-reagent/env/run.sh python3 /env/probe.py --checkpoint Qwen/Qwen3-1.7B
#
# Three mounts, each for a reason:
#   the Hugging Face cache   read-only at /hf, so loading a checkpoint cannot write to the host
#   a named Docker volume    writable at /work, so adapters and checkpoints have somewhere to go
#   this directory           read-only at /env, so the code that ran is the code in the repo
#   the repository           read-only at /repo, so the package under src/ is importable
#
# GPU access goes through CDI (`--device nvidia.com/gpu=all`). There is no `nvidia` runtime on
# this host, so `--gpus all` is not the flag that works here.
#
# The memory cap is not tidiness. On a unified-memory host (`MemTotal` = the GPU's memory) a trainer
# that holds too many autograd graphs does not fail in Python: the whole machine goes into swap, the
# kernel OOM killer runs, and because session processes carry `oom_score_adj` 100-200 while a trainer
# carries 0, the first thing it kills is the login session. That happened once - 2026-10-22, a
# 60-rollout round, ~70 GB of live graphs, both the vLLM server and the user's sshd-tied session
# killed with it. With `--memory`, the same mistake kills only this container.
#
# Note the caveat: `--ipc=host` means /dev/shm belongs to the host and is not charged to this
# container's cgroup. Nothing here uses shared memory between processes, but a DataLoader with
# workers would slip past the cap.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
image="${ARLH_TRAIN_IMAGE:-arlh-pytorch:26.09}"
hf_cache="${ARLH_HF_CACHE:-$HOME/.cache/huggingface}"
volume="${ARLH_VOLUME:-arlh-p2}"
mem="${ARLH_TRAIN_MEM:-48g}"
swap="${ARLH_TRAIN_SWAP:-$mem}"

[ "$#" -gt 0 ] || set -- python3 /env/probe.py

docker volume create "$volume" >/dev/null
exec docker run --rm -i \
  --memory="$mem" --memory-swap="$swap" \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/hf \
  -v "$hf_cache":/hf:ro \
  -v "$volume":/work \
  -v "$here":/env:ro \
  -v "$repo":/repo:ro \
  -e PYTHONPATH=/repo:/repo/p2-reagent/src \
  -w /work \
  "$image" "$@"
