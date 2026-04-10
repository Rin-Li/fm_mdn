from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.fm_utils import get_timesteps
from pfp.common.se3_utils import pfp_to_state5p_th, state5p_to_pfp_th


class FM5PPolicy(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        time_conditioning: bool,
        obs_encoder: nn.Module,
        diffusion_net: nn.Module,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        pos_emb_scale: int = 20,
        norm_pcd_center: list = None,
        noise_type: str = "gaussian",
        noise_scale: float = 1.0,
        loss_type: str = "l2",
        flow_schedule: str = "linear",
        exp_scale: float = None,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer
        self.time_conditioning = time_conditioning
        self.obs_encoder = obs_encoder
        self.diffusion_net = diffusion_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
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

    def _init_noise(self, batch_size: int) -> torch.Tensor:
        B = batch_size
        T = self.n_pred_steps
        noise_poses = pp.randn_SE3((B, T), device=DEVICE).matrix()
        noise_gripper = torch.randn((B, T, 1), device=DEVICE)
        noise_pfp = torch.cat(
            [
                noise_poses[..., :3, 3],
                noise_poses[..., :3, 0],
                noise_poses[..., :3, 1],
                noise_gripper,
            ],
            dim=-1,
        )
        noise_5p = pfp_to_state5p_th(noise_pfp)
        return noise_5p

    def _init_target(self, ny: torch.Tensor) -> torch.Tensor:
        """
        ny: (B, T, 10) -> xyz, rot6d, grip
        """
        target_5p = pfp_to_state5p_th(ny)
        return target_5p

    # ############### Training ################

    def forward(self, batch):
        """batch is the output of the dataloader"""
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
        loss_5p, loss_grip = self.calculate_loss(pcd, robot_state_obs, robot_state_pred)
        loss = self.l_w["5p"] * loss_5p + self.l_w["grip"] * loss_grip
        self.logger.log_metrics(
            {
                "loss/train/5p": loss_5p.item(),
                "loss/train/grip": loss_grip.item(),
            }
        )
        return loss

    def calculate_loss(
        self, pcd: torch.Tensor, robot_state_obs: torch.Tensor, robot_state_pred: torch.Tensor
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred

        B = ny.shape[0]
        T = ny.shape[1]

        # Sample random time step
        t_shape = [1] * len(ny.shape)
        t_shape[0] = ny.shape[0]  # B
        t = torch.rand(t_shape, device=DEVICE)

        # Initialize start and end poses + gripper state
        z0_5p = self._init_noise(B)
        z1_5p = self._init_target(ny)

        # Move to intermediate step
        z_t = t * z1_5p + (1.0 - t) * z0_5p

        # Calculate relative change between them
        target_vel = z1_5p - z0_5p

        # Do prediction
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None
        pred_vel = self.diffusion_net(z_t, timesteps, global_cond=nx)
        assert pred_vel.shape == (B, T, 16)

        # Calculate loss
        loss_5p = self.loss_fun(pred_vel[..., :15], target_vel[..., :15])
        loss_grip = self.loss_fun(pred_vel[..., 15], target_vel[..., 15])
        return loss_5p, loss_grip

    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch

        # Eval loss
        loss_5p, loss_grip = self.calculate_loss(pcd, robot_state_obs, robot_state_pred)
        loss_total = self.l_w["5p"] * loss_5p + self.l_w["grip"] * loss_grip
        self.logger.log_metrics(
            {
                "loss/eval/5p": loss_5p.item(),
                "loss/eval/grip": loss_grip.item(),
                "loss/eval/total": loss_total.item(),
            }
        )

        # Eval metrics
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
        B = nx.shape[0]
        z = self._init_noise(B) if noise is None else noise
        traj = [state5p_to_pfp_th(z)]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            timesteps = torch.ones((B), device=DEVICE) * t0[i]
            timesteps *= self.pos_emb_scale
            vel_pred = self.diffusion_net(z, timesteps, global_cond=nx)
            z = z.detach().clone() + vel_pred * dt[i]
            traj.append(state5p_to_pfp_th(z))

        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int,
        flow_schedule: str = None,
        exp_scale: float = None,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        # cfg.model.obs_encoder.encoder.random_crop = False
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: FM5PPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class FM5PPolicyImage(FM5PPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image
