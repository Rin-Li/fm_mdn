import hydra
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from composer.trainer import Trainer
from composer.loggers import WandBLogger
from composer.callbacks import LRMonitor
from composer.models import ComposerModel
from composer.algorithms import EMA
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from pfp import DEVICE, DATA_DIRS, set_seeds
from pfp.data.dataset_pcd import RobotDatasetPcd
from pfp.data.dataset_images import RobotDatasetImages


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)

    data_path_train = DATA_DIRS.PFP_REAL / cfg.task_name / "train"
    data_path_valid = DATA_DIRS.PFP_REAL / cfg.task_name / "valid"
    if cfg.obs_mode == "pcd":
        dataset_train = RobotDatasetPcd(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetPcd(data_path_valid, **cfg.dataset)
    elif cfg.obs_mode == "rgb":
        dataset_train = RobotDatasetImages(data_path_train, **cfg.dataset)
        dataset_valid = RobotDatasetImages(data_path_valid, **cfg.dataset)
    else:
        raise ValueError(f"Unknown observation mode: {cfg.obs_mode}")
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        **cfg.dataloader,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
    )

    composer_model: ComposerModel = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, composer_model.parameters())
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=(len(dataloader_train) * cfg.epochs),
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
    )

    wandb_logger = WandBLogger(
        project="pfp-real",
        entity="rl-lab-chisari",
        init_kwargs={
            "config": OmegaConf.to_container(cfg),
            "mode": "online" if cfg.log_wandb else "disabled",
        },
    )

    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=cfg.epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=[wandb_logger],
        callbacks=[LRMonitor()],
        save_folder="ckpt/{run_name}",
        save_interval=f"{cfg.save_each_n_epochs}ep",
        save_num_checkpoints_to_keep=3,
        algorithms=[EMA()] if cfg.use_ema else None,
        run_name=cfg.run_name,  # set this to continue training from previous ckpt
        autoresume=True if cfg.run_name is not None else False,
        spin_dataloaders=False,
    )
    wandb.watch(composer_model)
    # Save the used cfg for inference
    OmegaConf.save(cfg, "ckpt/" + trainer.state.run_name + "/config.yaml")

    trainer.fit()
    wandb.finish()
    trainer.close()
    return


if __name__ == "__main__":
    main()
