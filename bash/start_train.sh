#!/bin/bash

gpu_id=$1
task_name=$2
experiment=$3
tmux new-session -d -s "train_${gpu_id}_${task_name}_${experiment}"
tmux send-keys -t "train_${gpu_id}_${task_name}_${experiment}" "conda activate pfp_env && WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=$task_name +experiment=$experiment" Enter
