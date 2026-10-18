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
#
# GPU access goes through CDI (`--device nvidia.com/gpu=all`). There is no `nvidia` runtime on
# this host, so `--gpus all` is not the flag that works here.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${ARLH_TRAIN_IMAGE:-arlh-pytorch:26.09}"
hf_cache="${ARLH_HF_CACHE:-$HOME/.cache/huggingface}"
volume="${ARLH_VOLUME:-arlh-p2}"

[ "$#" -gt 0 ] || set -- python3 /env/probe.py

docker volume create "$volume" >/dev/null
exec docker run --rm -i \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/hf \
  -v "$hf_cache":/hf:ro \
  -v "$volume":/work \
  -v "$here":/env:ro \
  -w /work \
  "$image" "$@"
