from __future__ import annotations
import copy
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from composer.models import ComposerModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS


class DDIMPolicy(ComposerModel, BasePolicy):
    """Class to train the DDIM diffusion model"""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_train: int,
        num_k_infer: int,
        obs_encoder: nn.Module,
        diffusion_net: nn.Module,
        noise_scheduler_train: DDIMScheduler,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        norm_pcd_center: list = None,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.num_k_train = num_k_train
        self.num_k_infer = num_k_infer
        self.obs_encoder = obs_encoder
        self.diffusion_net = diffusion_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        # It's easier to have two different schedulers for training and eval/inference
        self.noise_scheduler_train = noise_scheduler_train
        self.noise_scheduler_infer = copy.deepcopy(noise_scheduler_train)
        self.noise_scheduler_infer.set_timesteps(num_k_infer)
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        self.noise_scheduler_infer.set_timesteps(num_k_infer)
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] += torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, robot_state_pred = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
        return pcd, robot_state_obs, robot_state_pred

    def _rand_range(self, low: float, high: float, size: tuple[int]) -> torch.Tensor:
        return torch.rand(size, device=DEVICE) * (high - low) + low

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, robot_state_pred = batch

        # xyz1 = self._rand_range(low=0.8, high=1.2, size=(3,))
        xyz2 = self._rand_range(low=-0.2, high=0.2, size=(3,))
        pcd[..., :3] = pcd[..., :3] + xyz2  # * xyz1 + xyz2
        robot_state_obs[..., :3] = robot_state_obs[..., :3] + xyz2  # * xyz1 + xyz2
        robot_state_pred[..., :3] = robot_state_pred[..., :3] + xyz2  # * xyz1 + xyz2

        # We shuffle the points, i.e. shuffle pcd along dim=2 (B, T, P, 3)
        idx = torch.randperm(pcd.shape[2])
        pcd = pcd[:, :, idx, :]
        return pcd, robot_state_obs, robot_state_pred

    # ########### TRAIN ###########

    def forward(self, batch):
        """batch: the output of the dataloader"""
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        outputs: the output of the forward pass
        batch: the output of the dataloader
        """
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch
        noise_pred, noise = self.train_noise(pcd, robot_state_obs, robot_state_pred)
        loss_xyz = nn.functional.mse_loss(noise_pred[..., :3], noise[..., :3])
        loss_rot6d = nn.functional.mse_loss(noise_pred[..., 3:9], noise[..., 3:9])
        loss_grip = nn.functional.mse_loss(noise_pred[..., 9], noise[..., 9])
        loss = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/train/xyz": loss_xyz.item(),
                "loss/train/rot6d": loss_rot6d.item(),
                "loss/train/grip": loss_grip.item(),
            }
        )
        return loss

    def train_noise(
        self, pcd: torch.Tensor, robot_state_obs: torch.Tensor, robot_state_pred: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred
        B = nx.shape[0]
        noise = torch.randn(ny.shape).to(DEVICE)
        timesteps = torch.randint(0, self.num_k_train, (B,)).long().to(DEVICE)
        noisy_y = self.noise_scheduler_train.add_noise(ny, noise, timesteps)
        noise_pred = self.diffusion_net(noisy_y, timesteps.float(), global_cond=nx)
        return noise_pred, noise

    # ########### EVAL ###########

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch
        pred_y = self.infer_y(pcd, robot_state_obs)
        mse_xyz = nn.functional.mse_loss(pred_y[..., :3], robot_state_pred[..., :3])
        mse_rot6d = nn.functional.mse_loss(pred_y[..., 3:9], robot_state_pred[..., 3:9])
        mse_grip = nn.functional.mse_loss(pred_y[..., 9], robot_state_pred[..., 9])
        self.logger.log_metrics(
            {
                "metrics/eval/mse_xyz": mse_xyz.item(),
                "metrics/eval/mse_rot6d": mse_rot6d.item(),
                "metrics/eval/mse_grip": mse_grip.item(),
            }
        )
        return pred_y

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        if noise is None:
            B = nx.shape[0]
            noise = torch.randn((B, *self.ny_shape), device=DEVICE)

        ny = noise
        traj = [ny]
        for k in self.noise_scheduler_infer.timesteps:
            noise_pred = self.diffusion_net(ny, k, global_cond=nx)
            if self.num_k_infer == 1:
                print("one step generation")
                ny = self.noise_scheduler_infer.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=ny,
                ).pred_original_sample
            else:
                ny = self.noise_scheduler_infer.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=ny,
                ).prev_sample
            traj.append(ny)
        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int = None,
        **kwargs,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: DDIMPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class DDIMPolicyImage(DDIMPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image


if __name__ == "__main__":
    ckpt_name = "1714199471-peculiar-earthworm"
    model = DDIMPolicy.load_from_checkpoint(ckpt_name, num_k_infer=10)
    print(model.obs_list)
