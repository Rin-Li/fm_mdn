# Just examples to copy paste in the tmux terminal

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=unplug_charger
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=unplug_charger model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=1 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=close_door
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=1 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=close_door model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=2 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_box
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=2 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_box model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=3 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_fridge
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=3 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_fridge model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=4 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=take_frame_off_hanger
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=4 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=take_frame_off_hanger model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=5 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_oven
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=5 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_oven model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=6 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=put_books_on_bookshelf
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=6 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=put_books_on_bookshelf model=flow_se3

# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=7 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=take_shoes_out_of_box
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=7 python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=take_shoes_out_of_box model=flow_se3

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=unplug_charger +experiment=pfp_so3,pfp_ddim,dp3,adaflow
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=unplug_charger +experiment=pfp_euclid,pfp_images,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=1 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=close_door +experiment=pfp_so3,pfp_ddim,dp3,adaflow
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=1 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=close_door +experiment=pfp_euclid,pfp_images,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=2 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=open_box +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=3 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=open_fridge +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=4 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=take_frame_off_hanger +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=5 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=open_oven +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=6 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=put_books_on_bookshelf +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=7 python scripts/train.py --multirun log_wandb=True dataloader.num_workers=8 task_name=take_shoes_out_of_box +experiment=pfp_so3,pfp_euclid,pfp_ddim,pfp_images,dp3,adaflow,diffusion_policy