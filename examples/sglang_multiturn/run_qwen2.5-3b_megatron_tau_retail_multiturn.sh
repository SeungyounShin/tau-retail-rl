# 8 × H100, tau_retail
set -x

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

ulimit -n 65535

CKPT="$HOME/models/qwen2.5-3b-mcore"
PROJECT_DIR=$(pwd)
CFG="$PROJECT_DIR/examples/sglang_multiturn/config"

python -m verl.trainer.main_ppo \
  --config-path "$CFG" \
  --config-name 'tau_retail_multiturn_megatron_grpo' \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=$CKPT \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=3 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  critic.ppo_max_token_len_per_gpu=4096 \
  critic.forward_max_token_len_per_gpu=4096 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=15 \
  trainer.project_name='tau_retail_async_rl' \
  trainer.experiment_name='qwen2.5-3b_tau_retail_tp4_pp2' \
  trainer.save_freq=-1 \
  trainer.test_freq=15 \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  data.train_files=$HOME/data/tau_retail/train.parquet \
  data.val_files=$HOME/data/tau_retail/test.parquet \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/tau_retail_tool_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/tau_retail_interaction_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5 "$@"
