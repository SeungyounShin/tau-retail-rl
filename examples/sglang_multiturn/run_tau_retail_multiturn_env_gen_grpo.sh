# run on 8xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

max_prompt_length=4096
max_response_length=10240

use_dynamic_bsz=True
ref_offload=True

actor_offload=False
gen_tp=2
fsdp_size=8

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4))
critic_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4))

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='tau_retail_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_batch_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tau_retail_async_rl' \
    trainer.experiment_name='qwen_tau_retail_multiturn_train_split' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.log_val_generations=False \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    critic.ppo_max_token_len_per_gpu=${critic_max_token_len_per_gpu} \
    critic.forward_max_token_len_per_gpu=${critic_max_token_len_per_gpu} \
    data.train_files=$HOME/data/tau_retail/train.parquet \
    data.val_files=$HOME/data/tau_retail/test.parquet \
    trainer.validation_data_dir="$PROJECT_DIR/val_generations" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/tau_retail_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/tau_retail_interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=25 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=25 $@