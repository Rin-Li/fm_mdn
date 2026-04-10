import hydra
import subprocess
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from composer.trainer import Trainer
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from pfp import DEVICE, REPO_DIRS, DATA_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd
from pfp.data.dataset_images import RobotDatasetImages


@hydra.main(version_base=None, config_path="../conf", config_name="trainer_eval")
def main(cfg: OmegaConf):
    # Download checkpoint if not present
    ckpt_path = REPO_DIRS.CKPT / cfg.run_name
    if not ckpt_path.exists():
        subprocess.run(
            [
                "rsync",
                "-hPrl",
                f"chisari@rlgpu2:{ckpt_path}",
                f"{REPO_DIRS.CKPT}/",
            ]
        )

    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    cfg = OmegaConf.merge(train_cfg, cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    data_path_valid = DATA_DIRS.PFP / cfg.task_name / "valid"
    if cfg.obs_mode == "pcd":
        dataset_valid = RobotDatasetPcd(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        dataset_valid = RobotDatasetImages(data_path_valid, **cfg.dataset)
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")
    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    composer_model: ComposerModel = hydra.utils.instantiate(cfg.model)
    wandb_logger = WandBLogger(
        project="pfp-trainer-eval",
        entity="rl-lab-chisari",
        init_kwargs={
            "config": OmegaConf.to_container(cfg),
            "mode": "online" if cfg.log_wandb else "disabled",
        },
    )

    trainer = Trainer(
        model=composer_model,
        eval_dataloader=dataloader_valid,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=[wandb_logger],
        save_folder="ckpt/{run_name}",
        run_name=cfg.run_name,  # set this to continue training from previous ckpt
        autoresume=True if cfg.run_name is not None else False,
    )
    trainer.eval()
    return


if __name__ == "__main__":
    main()
