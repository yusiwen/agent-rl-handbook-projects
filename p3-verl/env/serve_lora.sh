#!/usr/bin/env bash
# Serve the base policy with LoRA enabled, so a training round can push its adapter into the
# running server instead of restarting it.
#
#   bash p3-verl/env/serve_lora.sh            # start
#   bash p3-verl/env/serve_lora.sh --stop     # remove
#
# Why this is the shape of the p3 demo, measured on the Spark on 2026-10-20:
#
#   restart the server with new merged weights   168 s   (process start -> /health 200)
#   load one adapter over HTTP                    52 ms   (25.7 MB, same base model resident)
#
# Restarting also costs the KV cache, the CUDA graph capture and whatever the previous round had
# warmed. Loading an adapter costs neither: the base weights never move, and each round registers
# its cumulative adapter under a new name, which is the smallest honest version of weight sync.
#
# Three details, each of which cost a restart to find:
#
# 1. `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` is required. Without it the server accepts `--enable-lora`
#    and still answers 404 on /v1/load_lora_adapter, because the runtime routes are gated on a
#    separate flag.
# 2. The adapter path is resolved *inside* the container. `/models` is the named volume that the
#    training container mounts at `/work`, so the trainer writes `/work/rl-c/adapter-r6` and the
#    server reads `/models/rl-c/adapter-r6`. Same bytes, two names.
# 3. `--max-lora-rank` must be at least the rank the adapter was trained with (16 here). A smaller
#    value fails the load with a shape error, not with a suggestion.
set -euo pipefail

NAME=${NAME:-p3-sync}
PORT=${PORT:-8003}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
BASE_NAME=${BASE_NAME:-qwen3-1.7b-base}
HF_CACHE=${HF_CACHE:-/home/yusiwen/.cache/huggingface}
VLLM_CACHE=${VLLM_CACHE:-/home/yusiwen/.cache/vllm}
VOLUME=${VOLUME:-arlh-p2}

if [ "${1:-}" = "--stop" ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "removed $NAME"
  exit 0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --ipc=host -p "${PORT}:8000" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${VOLUME}:/models" \
  "$IMAGE" "$MODEL" \
  --served-model-name "$BASE_NAME" \
  --enable-lora --max-lora-rank 16 \
  --gpu-memory-utilization 0.10 --max-model-len 8192 >/dev/null

echo "waiting for http://127.0.0.1:${PORT}/health ..."
start=$(date +%s%N)
for _ in $(seq 1 120); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)" = "200" ]; then
    end=$(date +%s%N)
    echo "healthy after $(( (end - start) / 1000000 )) ms"
    echo "base model: $BASE_NAME  (adapters are loaded with: POST /v1/load_lora_adapter)"
    exit 0
  fi
  sleep 3
done
echo "FAILED: no healthy response after 6 minutes; docker logs $NAME" >&2
exit 1
