#!/usr/bin/env bash
# GRPO learning check: 24 calibrated prompts, n=6, 12 steps, final checkpoint saved.
# Second pass at the memory budget: the first reached step 4 and was OOM-killed with no traceback
# (the main process was SIGKILLed), so the rollout batch, the micro-batches and the vLLM fraction
# are all smaller here.  Eight epochs so that 12 steps of 16 prompts have data to draw from.
#
# In-loop validation is off on purpose: a second dataloader pushed the container past its memory
# ceiling and the dataloader was killed. The held-out number is measured separately against the
# saved checkpoint, so that the pre and post numbers use the same protocol.
# Run on the Spark with the :8003 server stopped - the loop held 93 GB at step 2 of the smoke run.
set -euo pipefail
docker run --rm --device nvidia.com/gpu=all --ipc=host --shm-size=16g --memory=118g --memory-swap=118g \
  -e HOME=/work -e TORCHINDUCTOR_CACHE_DIR=/work/tcache -e HF_HOME=/hf -e HF_DATASETS_CACHE=/work/hf-datasets \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_V1=1 -e RAY_DEDUP_LOGS=1 \
  -v "$HOME/arlh/p3c:/work" -v "$HOME/.cache/huggingface:/hf:ro" \
  --entrypoint bash vllm/vllm-openai:v0.24.0 -lc '
    /work/env/bin/python -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
      data.train_files=/work/train.parquet data.val_files=/work/eval.parquet \
      data.train_batch_size=16 data.val_batch_size=8 data.dataloader_num_workers=0 \
      data.max_prompt_length=256 data.max_response_length=384 \
      data.filter_overlong_prompts=False \
      +data.apply_chat_template_kwargs.enable_thinking=false \
      actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=6 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
      actor_rollout_ref.rollout.max_model_len=768 \
      actor_rollout_ref.rollout.response_length=384 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.rollout.free_cache_engine=True \
      actor_rollout_ref.rollout.enforce_eager=True \
      trainer.val_before_train=False trainer.test_freq=8 \
      trainer.save_freq=8 trainer.default_local_dir=/work/ckpt \
      trainer.total_epochs=8 trainer.total_training_steps=8 \
      trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.logger="[console]"
  ' > "$HOME/arlh/p3c/grpo_train.log" 2>&1
echo "container exit: $?"
