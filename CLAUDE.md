# CLAUDE.md

## Environment

Always use the project's virtual environment:

```bash
source /home/kklab-ur-robot/fm_mdn/.venv/bin/activate
```

All Python commands must be run with this venv active. The venv contains all dependencies (torch, composer, hydra, pfp, etc.).

## Project Root

Working directory: `/home/kklab-ur-robot/fm_mdn`

## Training Commands

### Stage 1: Train SO3 Prior

```bash
source .venv/bin/activate
python scripts/train_so3_prior.py \
  task_name=take_frame_off_hanger \
  run_name=so3_prior_take_frame \
  log_wandb=False \
  dataloader.num_workers=8
```

### Stage 2: Train FM with learned SO3 init

```bash
source .venv/bin/activate
python scripts/train.py \
  task_name=take_frame_off_hanger \
  +experiment=pointflowmatch_so3_learned \
  model.so3_prior_ckpt_name=so3_prior_take_frame \
  log_wandb=False \
  dataloader.num_workers=8
```

## Data

Training data is at `demos/sim/<task_name>/{train,valid}/` in Zarr format.

Currently available tasks: `take_frame_off_hanger`
