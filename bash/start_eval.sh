#!/bin/bash

gpu_id=$1
ckpt_name=$2
k_steps=${3:-50}  # Default k_steps is 50
seed=${4:-0}  # Default seed is 0
if [ $seed -ne 0 ]; then
    tmux new-session -d -s "eval_${ckpt_name}_k${k_steps}_${seed}" 
    tmux send-keys -t "eval_${ckpt_name}_k${k_steps}_${seed}" "conda activate pfp_env && CUDA_VISIBLE_DEVICES=$gpu_id WANDB__SERVICE_WAIT=300 xvfb-run -a python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=False policy.ckpt_name=$ckpt_name seed=$seed policy.num_k_infer=$k_steps" Enter
else
    for seed in 5678 2468 1357; do
        tmux new-session -d -s "eval_${ckpt_name}_k${k_steps}_${seed}"
        tmux send-keys -t "eval_${ckpt_name}_k${k_steps}_${seed}" "conda activate pfp_env && CUDA_VISIBLE_DEVICES=$gpu_id WANDB__SERVICE_WAIT=300 xvfb-run -a python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=False policy.ckpt_name=$ckpt_name seed=$seed policy.num_k_infer=$k_steps" Enter
    done
fi