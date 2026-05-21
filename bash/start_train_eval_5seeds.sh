#!/bin/bash
set -euo pipefail

gpu_id=${1:-4}
task_name=${2:-take_frame_off_hanger}
experiment=${3:-tube_local_so3_flow}
k_steps=${4:-50}
num_seeds=${5:-5}
run_name=${6:-"${task_name}_${experiment}_$(date +%Y%m%d_%H%M%S)"}
log_wandb=${PFP_LOG_WANDB:-True}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_path="${repo_dir}/bash/$(basename "${BASH_SOURCE[0]}")"
session_name="train_eval_${gpu_id}_${task_name}_${experiment}_${run_name}"
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
        "PFP_TRAIN_EVAL_INSIDE=1 PFP_TRAIN_EVAL_LOG='${log_path}' bash '${script_path}' '$gpu_id' '$task_name' '$experiment' '$k_steps' '$num_seeds' '$run_name'"
    echo "Started tmux session: ${session_name}"
    echo "Run name: ${run_name}"
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

echo "Repo: ${repo_dir}"
echo "Task data: ${repo_dir}/demos/sim/${task_name}"
echo "Experiment: ${experiment}"
echo "Run name: ${run_name}"
echo "K steps: ${k_steps}"
echo "Eval seeds: ${num_seeds} random seeds"
echo "W&B logging: ${log_wandb}"

python scripts/train.py \
    log_wandb="${log_wandb}" \
    dataloader.num_workers=8 \
    task_name="${task_name}" \
    run_name="${run_name}" \
    auto_eval=False \
    +experiment="${experiment}"

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
    echo "Evaluating ${run_name} with seed=${seed}"
    xvfb-run -a python scripts/evaluate.py \
        log_wandb="${log_wandb}" \
        env_runner.env_config.vis=False \
        policy.ckpt_name="${run_name}" \
        seed="${seed}" \
        policy.num_k_infer="${k_steps}"
done

echo "Train + ${num_seeds} eval seeds finished for ${run_name}"
