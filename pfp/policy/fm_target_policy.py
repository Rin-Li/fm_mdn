from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import pfp_to_pose_th
from pfp.common.fm_utils import get_timesteps


class FMTargetPolicy(ComposerModel, BasePolicy):
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
        augment_data: bool,
        loss_weights: dict[int],
        norm_pcd_center: list,
        loss_type: str,
        pos_emb_scale: int = 20,
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

    def _init_noise(self, batch_size: int) -> tuple[torch.Tensor, pp.SO3, torch.Tensor]:
        B = batch_size
        T = self.n_pred_steps
        noise_xyz = torch.randn((B, T, 3), device=DEVICE)
        noise_SO3 = pp.randn_SO3((B, T), device=DEVICE)
        noise_gripper = torch.randn((B, T, 1), device=DEVICE)
        return noise_xyz, noise_SO3, noise_gripper

    def _pfp_to_pp(self, pfp_state: torch.Tensor) -> tuple[pp.SE3, torch.Tensor]:
        """
        pfp_state: (B, T, 10) -> xyz, rot6d, grip
        """
        poses_th, gripper_th = pfp_to_pose_th(pfp_state)  # (B, T, 4, 4)
        xyz = poses_th[..., :3, 3]
        rot_SO3 = pp.mat2SO3(poses_th[..., :3, :3], check=False)  # (B, T, 4)
        gripper = gripper_th
        return xyz, rot_SO3, gripper

    def _pp_to_pfp(
        self, z_xyz: torch.Tensor, z_SO3: pp.SO3, z_gripper: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_xyz: (B, T, 3) xyz
            z_SO3: (B, T, 4) pp.SO3 rotation
            z_gripper: (B, T, 1) gripper
        Returns:
            z: (B, T, 10) pfp state
        """
        B, T, _ = z_xyz.shape
        z = torch.zeros((B, T, 10), device=DEVICE)
        rot = pp.matrix(z_SO3)
        z[..., :3] = z_xyz
        z[..., 3:9] = rot[..., :3, :2].mT.flatten(start_dim=-2)
        z[..., 9:] = z_gripper
        return z

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
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred
        )
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

    def calculate_loss(
        self, pcd: torch.Tensor, robot_state_obs: torch.Tensor, robot_state_pred: torch.Tensor
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs)
        ny: torch.Tensor = robot_state_pred

        B = ny.shape[0]
        # T = ny.shape[1]

        # Sample random time step
        t_shape = (B, 1, 1)
        t = torch.rand(t_shape, device=DEVICE)

        # Initialize start and end poses + gripper state
        z0_xyz, z0_SO3, z0_gripper = self._init_noise(B)
        z1_xyz, z1_SO3, z1_gripper = self._pfp_to_pp(ny)
        target_pfp = ny

        # Calculate relative change between them
        target_vel_xyz = z1_xyz - z0_xyz
        target_vel_so3 = pp.Log(pp.Inv(z0_SO3) @ z1_SO3)
        target_vel_gripper = z1_gripper - z0_gripper

        # Move to intermediate step
        zt_xyz = z0_xyz + target_vel_xyz * t
        zt_SO3: pp.SO3 = z0_SO3 @ pp.Exp(target_vel_so3 * t)
        zt_gripper: torch.Tensor = z0_gripper + target_vel_gripper * t

        # Convert to pfp network input representation
        zt_pfp = self._pp_to_pfp(zt_xyz, zt_SO3, zt_gripper)
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None

        # Do prediction
        pred_pfp = self.diffusion_net(zt_pfp, timesteps, global_cond=nx)
        assert pred_pfp.shape == zt_pfp.shape
        # TODO: you could do procrustes here

        # Calculate loss
        loss_xyz = self.loss_fun(pred_pfp[..., :3], target_pfp[..., :3])
        loss_rot6d = self.loss_fun(pred_pfp[..., 3:9], target_pfp[..., 3:9])
        loss_grip = self.loss_fun(pred_pfp[..., 9], target_pfp[..., 9])
        return loss_xyz, loss_rot6d, loss_grip

    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch

        # Eval loss
        loss_xyz, loss_rot6d, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred
        )
        loss_total = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w["rot6d"] * loss_rot6d
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/eval/xyz": loss_xyz.item(),
                "loss/eval/rot6d": loss_rot6d.item(),
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
        z_xyz, z_SO3, z_gripper = self._init_noise(B) if noise is None else noise
        z = self._pp_to_pfp(z_xyz, z_SO3, z_gripper)
        traj = [z]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            t = torch.ones((B), device=DEVICE) * t0[i]
            timesteps = t * self.pos_emb_scale if self.time_conditioning else None
            pred_final_pfp = self.diffusion_net(z, timesteps, global_cond=nx)
            z1_xyz, z1_SO3, z1_gripper = self._pfp_to_pp(pred_final_pfp)

            z_xyz = z_xyz + (z1_xyz - z_xyz) * dt[i]
            z_SO3 = z_SO3 @ pp.Exp(pp.Log(pp.Inv(z_SO3) @ z1_SO3) * dt[i])
            z_gripper = z_gripper + (z1_gripper - z_gripper) * dt[i]

            z = self._pp_to_pfp(z_xyz, z_SO3, z_gripper)
            traj.append(z)
        return torch.stack(traj) if return_traj else traj[-1]

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
        model: FMTargetPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class FMTargetPolicyImage(FMTargetPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image
