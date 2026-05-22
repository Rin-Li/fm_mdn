#!/bin/bash
set -euo pipefail

gpu_id=${1:-4}
task_name=${2:-take_frame_off_hanger}
k_steps=${3:-50}
num_seeds=${4:-5}
stage1_epochs=${5:-500}
stage2_epochs=${6:-1500}
run_prefix=${7:-"${task_name}_shared_encoder_tube_flow_$(date +%Y%m%d_%H%M%S)"}
log_wandb=${PFP_LOG_WANDB:-True}
debug_stats=${PFP_DEBUG_STATS:-False}
debug_stats_interval=${PFP_DEBUG_STATS_INTERVAL:-1}
stage1_use_ema=${PFP_STAGE1_USE_EMA:-False}
stage2_use_ema=${PFP_STAGE2_USE_EMA:-True}
stage1_freeze_encoder=${PFP_STAGE1_FREEZE_ENCODER:-True}
stage1_lr=${PFP_STAGE1_LR:-1.0e-4}
stage2_lr=${PFP_STAGE2_LR:-3.0e-5}
stage1_warmup=${PFP_STAGE1_WARMUP:-0}
stage2_warmup=${PFP_STAGE2_WARMUP:-5000}
stage1_save_each=${PFP_STAGE1_SAVE_EACH:-50}
stage2_save_each=${PFP_STAGE2_SAVE_EACH:-100}
n_points_override=${PFP_N_POINTS:-}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_path="${repo_dir}/bash/$(basename "${BASH_SOURCE[0]}")"
session_name="shared_tube_2stage_${gpu_id}_${run_prefix}"
session_name="${session_name//[^A-Za-z0-9_]/_}"
log_path="${repo_dir}/logs/${session_name}.log"

if [[ "${PFP_TRAIN_EVAL_INSIDE:-0}" == "1" && -n "${PFP_TRAIN_EVAL_LOG:-}" ]]; then
    mkdir -p "$(dirname "${PFP_TRAIN_EVAL_LOG}")"
    exec > >(tee -a "${PFP_TRAIN_EVAL_LOG}") 2>&1
    trap 'status=$?; if [[ ${status} -ne 0 ]]; then echo "Failed with exit code ${status}. Log: ${PFP_TRAIN_EVAL_LOG}"; [[ -n "${TMUX:-}" ]] && exec bash; fi' EXIT
fi

if [[ "${PFP_TRAIN_EVAL_INSIDE:-0}" != "1" ]] \
    && [[ "${PFP_NO_TMUX:-0}" != "1" ]] \
    && command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${session_name}" \
        "PFP_TRAIN_EVAL_INSIDE=1 PFP_TRAIN_EVAL_LOG='${log_path}' bash '${script_path}' '$gpu_id' '$task_name' '$k_steps' '$num_seeds' '$stage1_epochs' '$stage2_epochs' '$run_prefix'"
    echo "Started tmux session: ${session_name}"
    echo "Stage 1 run: ${run_prefix}_tube"
    echo "Stage 2 run: ${run_prefix}_flow"
    echo "Log file: ${log_path}"
    echo "Attach with: tmux attach -t ${session_name}"
    exit 0
fi

if [[ "${PFP_TRAIN_EVAL_INSIDE:-0}" != "1" ]]; then
    echo "tmux not found or disabled; running train/eval in the current shell."
fi

cd "${repo_dir}"
if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
    source "${VIRTUAL_ENV}/bin/activate"
    echo "Using active virtualenv: ${VIRTUAL_ENV}"
elif [[ -f "${repo_dir}/.venv/bin/activate" ]]; then
    source "${repo_dir}/.venv/bin/activate"
    echo "Using repo virtualenv: ${repo_dir}/.venv"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${PFP_CONDA_ENV:-pfp_env}"
    echo "Using conda env: ${PFP_CONDA_ENV:-pfp_env}"
else
    echo "No conda or active virtualenv detected; using current python."
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB__SERVICE_WAIT=300

stage1_run="${run_prefix}_tube"
stage2_run="${run_prefix}_flow"

echo "Repo: ${repo_dir}"
echo "Task data: ${repo_dir}/demos/sim/${task_name}"
echo "Stage 1 run: ${stage1_run}"
echo "Stage 2 run: ${stage2_run}"
echo "Stage 1 epochs: ${stage1_epochs}"
echo "Stage 2 epochs: ${stage2_epochs}"
echo "Save intervals: stage1=${stage1_save_each} epochs, stage2=${stage2_save_each} epochs"
echo "K steps: ${k_steps}"
echo "Eval seeds: ${num_seeds} random seeds"
echo "W&B logging: ${log_wandb}"
echo "Debug stats: ${debug_stats}, interval=${debug_stats_interval}"
echo "EMA: stage1=${stage1_use_ema}, stage2=${stage2_use_ema}"
echo "Stage 1 freeze encoder: ${stage1_freeze_encoder}"
echo "Learning rates: stage1=${stage1_lr}, stage2=${stage2_lr}"
echo "Warmup steps: stage1=${stage1_warmup}, stage2=${stage2_warmup}"
if [[ -n "${n_points_override}" ]]; then
    echo "Point cloud points override: ${n_points_override}"
fi

stage1_overrides=(
    log_wandb="${log_wandb}"
    dataloader.num_workers=8
    task_name="${task_name}"
    run_name="${stage1_run}"
    epochs="${stage1_epochs}"
    save_each_n_epochs="${stage1_save_each}"
    use_ema="${stage1_use_ema}"
    optimizer.lr="${stage1_lr}"
    lr_scheduler.num_warmup_steps="${stage1_warmup}"
    auto_eval=False
    model.freeze_obs_encoder="${stage1_freeze_encoder}"
    model.debug_stats="${debug_stats}"
    model.debug_stats_interval="${debug_stats_interval}"
    +experiment=shared_encoder_tube_local_flow
)
stage2_overrides=(
    log_wandb="${log_wandb}"
    dataloader.num_workers=8
    task_name="${task_name}"
    run_name="${stage2_run}"
    epochs="${stage2_epochs}"
    save_each_n_epochs="${stage2_save_each}"
    use_ema="${stage2_use_ema}"
    optimizer.lr="${stage2_lr}"
    lr_scheduler.num_warmup_steps="${stage2_warmup}"
    auto_eval=False
    model.debug_stats="${debug_stats}"
    model.debug_stats_interval="${debug_stats_interval}"
    +experiment=shared_encoder_tube_local_flow_stage2
    model.init_ckpt_name="${stage1_run}"
    model.init_ckpt_episode=latest-rank0.pt
)
if [[ -n "${n_points_override}" ]]; then
    stage1_overrides+=(dataset.n_points="${n_points_override}")
    stage2_overrides+=(dataset.n_points="${n_points_override}")
fi

python scripts/train.py "${stage1_overrides[@]}"

python scripts/train.py "${stage2_overrides[@]}"

mapfile -t seeds < <(
    python - "${num_seeds}" <<'PY'
import random
import sys

n = int(sys.argv[1])
for seed in random.sample(range(1000, 999999), n):
    print(seed)
PY
)

echo "Running eval seeds: ${seeds[*]}"
for seed in "${seeds[@]}"; do
    echo "Evaluating ${stage2_run} with seed=${seed}"
    xvfb-run -a python scripts/evaluate.py \
        log_wandb="${log_wandb}" \
        env_runner.env_config.vis=False \
        policy.ckpt_name="${stage2_run}" \
        seed="${seed}" \
        policy.num_k_infer="${k_steps}"
done

echo "Shared encoder tube flow two-stage train + ${num_seeds} eval seeds finished."
