from __future__ import annotations

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from omegaconf import OmegaConf

from pfp import DEVICE, REPO_DIRS
from pfp.common.fm_utils import get_timesteps
from pfp.common.smooth_noise import clamp_norm, sample_residual_y0
from pfp.common.so3 import (
    decode_residual,
    encode_residual,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_log,
)
from pfp.data.dataset_pcd import augment_pcd_data
from pfp.policy.base_policy import BasePolicy


class TubeLocalFlowPolicy(ComposerModel, BasePolicy):
    """Tube-conditioned local SO(3) residual flow policy."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        time_conditioning: bool,
        obs_encoder: nn.Module,
        tube_predictor: nn.Module,
        diffusion_net: nn.Module,
        augment_data: bool = False,
        loss_weights: dict[str, float] | None = None,
        pos_emb_scale: int = 20,
        norm_pcd_center: list[float] | None = None,
        loss_type: str = "l2",
        flow_schedule: str = "linear",
        exp_scale: float | None = None,
        snr_sampler: str = "uniform",
        training_stage: str = "tube",
        include_gripper: bool = True,
        freeze_tube_predictor: bool = False,
        tube_radius: float = 1.0471975512,
        residual_radius_factor: float = 0.8,
        n_ctrl: int = 8,
        sigma_p: float = 0.03,
        sigma_r: float = 0.15,
        sigma_g: float = 0.1,
        inference_init: str = "zero",
        flow_gripper_mode: str = "last_obs",
        init_ckpt_name: str | None = None,
        init_ckpt_episode: str = "latest-rank0.pt",
        freeze_obs_encoder: bool = False,
        debug_stats: bool = False,
        debug_stats_interval: int = 1,
        subs_factor: int = 1,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps, subs_factor)
        if training_stage not in ["tube", "flow", "joint"]:
            raise ValueError(f"Unknown training_stage: {training_stage}")
        if inference_init not in ["zero", "smooth_noise"]:
            raise ValueError(f"Unknown inference_init: {inference_init}")
        if flow_gripper_mode not in ["last_obs", "zero"]:
            raise ValueError(f"Unknown flow_gripper_mode: {flow_gripper_mode}")

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer
        self.time_conditioning = time_conditioning
        self.obs_encoder = obs_encoder
        self.tube_predictor = tube_predictor
        self.diffusion_net = diffusion_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights or {}
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        self.snr_sampler = snr_sampler
        self.training_stage = training_stage
        self.include_gripper = include_gripper
        self.freeze_tube_predictor = freeze_tube_predictor
        self.tube_radius = tube_radius
        self.residual_radius = residual_radius_factor * tube_radius
        self.n_ctrl = n_ctrl
        self.sigma_p = sigma_p
        self.sigma_r = sigma_r
        self.sigma_g = sigma_g
        self.inference_init = inference_init
        self.flow_gripper_mode = flow_gripper_mode
        self.init_ckpt_name = init_ckpt_name
        self.init_ckpt_episode = init_ckpt_episode
        self.freeze_obs_encoder = freeze_obs_encoder
        self.debug_stats = debug_stats
        self.debug_stats_interval = debug_stats_interval
        self._debug_step = 0
        self.residual_dim = 7 if include_gripper else 6

        if init_ckpt_name is not None:
            self._load_shared_encoder_tube(init_ckpt_name, init_ckpt_episode)

        if freeze_tube_predictor:
            for param in self.tube_predictor.parameters():
                param.requires_grad_(False)
        if freeze_obs_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad_(False)

        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError

    def _load_shared_encoder_tube(self, ckpt_name: str, ckpt_episode: str) -> None:
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        state_dict = torch.load(ckpt_path_list[0], map_location=DEVICE)["state"]["model"]
        own_state = self.state_dict()
        shared_state = {
            key: value
            for key, value in state_dict.items()
            if (key.startswith("obs_encoder.") or key.startswith("tube_predictor."))
            and key in own_state
            and own_state[key].shape == value.shape
        }
        missing = self.load_state_dict(shared_state, strict=False)
        if len(shared_state) == 0:
            raise RuntimeError(f"No shared encoder/tube weights loaded from {ckpt_path_list[0]}")
        unexpected = list(missing.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading shared weights: {unexpected}")

    def _debug_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        tensor_detached = tensor.detach()
        finite = torch.isfinite(tensor_detached)
        finite_count = finite.sum().item()
        total_count = tensor_detached.numel()
        if finite_count == 0:
            print(f"[tube-debug] {name}: shape={tuple(tensor.shape)} finite=0/{total_count}")
            return
        vals = tensor_detached[finite].float()
        print(
            "[tube-debug] "
            f"{name}: shape={tuple(tensor.shape)} finite={finite_count}/{total_count} "
            f"min={vals.min().item():.6g} max={vals.max().item():.6g} "
            f"mean={vals.mean().item():.6g} std={vals.std(unbiased=False).item():.6g}"
        )

    def _debug_policy_stats(
        self,
        tag: str,
        tensors: dict[str, torch.Tensor],
        force: bool = False,
    ) -> None:
        if not self.debug_stats:
            return
        interval = max(int(self.debug_stats_interval), 1)
        if not force and self._debug_step % interval != 0:
            return
        print(f"[tube-debug] step={self._debug_step} tag={tag}")
        for name, tensor in tensors.items():
            if tensor is not None:
                self._debug_tensor_stats(name, tensor)

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=pcd.device)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=robot_state.device)
        robot_state[..., 9] -= torch.tensor(0.5, device=robot_state.device)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=robot_state.device)
        robot_state[..., 9] += torch.tensor(0.5, device=robot_state.device)
        return robot_state

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, robot_state_pred = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
        return pcd, robot_state_obs, robot_state_pred

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return augment_pcd_data(batch)

    def _sample_snr(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.snr_sampler == "uniform":
            return torch.rand((batch_size, 1, 1), device=device)
        if self.snr_sampler == "logit_normal":
            return torch.sigmoid(torch.randn((batch_size, 1, 1), device=device))
        raise NotImplementedError

    def _pfp_to_pose(
        self, pfp_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            pfp_state[..., :3],
            rotation_6d_to_matrix(pfp_state[..., 3:9]),
            pfp_state[..., 9:10],
        )

    def _pose_to_pfp(
        self,
        pos: torch.Tensor,
        rot: torch.Tensor,
        gripper: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros((*pos.shape[:-1], self.y_dim), dtype=pos.dtype, device=pos.device)
        out[..., :3] = pos
        out[..., 3:9] = matrix_to_rotation_6d(rot)
        out[..., 9:] = gripper
        return out

    def _predict_tube(
        self,
        cond: torch.Tensor,
        detach_tube: bool = False,
    ) -> dict[str, torch.Tensor]:
        if detach_tube:
            with torch.no_grad():
                pred = self.tube_predictor(cond)
            return {key: value.detach() for key, value in pred.items()}
        return self.tube_predictor(cond)

    def _mu_feature(self, tube: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [tube["mu_p"], matrix_to_rotation_6d(tube["mu_R"])]
        if self.include_gripper:
            parts.append(tube["mu_g"])
        return torch.cat(parts, dim=-1)

    def _target_residual(
        self,
        pos: torch.Tensor,
        rot: torch.Tensor,
        gripper: torch.Tensor,
        tube: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        residual = encode_residual(pos, rot, tube["mu_p"], tube["mu_R"])
        if self.include_gripper:
            residual = torch.cat([residual, gripper - tube["mu_g"]], dim=-1)
        return residual

    def _decode_action(
        self,
        residual: torch.Tensor,
        tube: dict[str, torch.Tensor],
        robot_state_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos, rot = decode_residual(residual[..., :6], tube["mu_p"], tube["mu_R"])
        if self.include_gripper:
            gripper = tube["mu_g"] + residual[..., 6:7]
        elif self.flow_gripper_mode == "last_obs" and robot_state_obs is not None:
            gripper = robot_state_obs[:, -1:, 9:10].expand(-1, residual.shape[1], -1)
        else:
            gripper = torch.zeros((*residual.shape[:-1], 1), device=residual.device)
        return self._pose_to_pfp(pos, rot, gripper)

    def _sample_base_residual(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return sample_residual_y0(
            batch_size=batch_size,
            horizon=self.n_pred_steps,
            sigma_p=self.sigma_p,
            sigma_r=self.sigma_r,
            r0=self.residual_radius,
            n_ctrl=self.n_ctrl,
            device=device,
            include_gripper=self.include_gripper,
            sigma_g=self.sigma_g,
        )

    def _zero_base_residual(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.n_pred_steps, self.residual_dim),
            dtype=torch.float32,
            device=device,
        )

    def _tube_losses(
        self,
        pos: torch.Tensor,
        rot: torch.Tensor,
        gripper: torch.Tensor,
        tube: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        xi = so3_log(tube["mu_R"].transpose(-1, -2) @ rot)
        xi_norm = xi.norm(dim=-1)
        zero = pos.new_zeros(())

        if self.n_pred_steps > 2:
            smooth_p = (
                tube["mu_p"][:, 2:] - 2.0 * tube["mu_p"][:, 1:-1] + tube["mu_p"][:, :-2]
            ).pow(2).mean()
            omega = so3_log(tube["mu_R"][:, :-1].transpose(-1, -2) @ tube["mu_R"][:, 1:])
            smooth_r = (omega[:, 1:] - omega[:, :-1]).pow(2).mean()
        else:
            smooth_p = zero
            smooth_r = zero

        losses = {
            "center_xyz": self.loss_fun(tube["mu_p"], pos),
            "center_so3": xi.pow(2).mean(),
            "chart": F.relu(xi_norm - self.tube_radius).pow(2).mean(),
            "chart_violation_rate": (xi_norm > self.tube_radius).float().mean(),
            "smooth_xyz": smooth_p,
            "smooth_so3": smooth_r,
        }
        if self.include_gripper:
            losses["center_grip"] = self.loss_fun(tube["mu_g"], gripper)
        else:
            losses["center_grip"] = zero
        return losses

    def _flow_losses(
        self,
        cond: torch.Tensor,
        pos: torch.Tensor,
        rot: torch.Tensor,
        gripper: torch.Tensor,
        tube: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = pos.shape[0]
        device = pos.device
        y1 = self._target_residual(pos, rot, gripper, tube)
        y0 = self._sample_base_residual(batch_size, device)
        s = self._sample_snr(batch_size, device)
        ys = y0 + s * (y1 - y0)
        target_vel = y1 - y0

        flow_input = torch.cat([ys, self._mu_feature(tube)], dim=-1)
        timesteps = s.view(batch_size) * self.pos_emb_scale if self.time_conditioning else None
        pred_vel = self.diffusion_net(flow_input, timesteps, global_cond=cond)
        if pred_vel.shape != target_vel.shape:
            raise RuntimeError(f"Expected flow output {target_vel.shape}, got {pred_vel.shape}")

        xi_norm = y1[..., 3:6].norm(dim=-1)
        losses = {
            "fm_xyz": self.loss_fun(pred_vel[..., :3], target_vel[..., :3]),
            "fm_so3": self.loss_fun(pred_vel[..., 3:6], target_vel[..., 3:6]),
            "chart_fm": F.relu(xi_norm - self.tube_radius).pow(2).mean(),
        }
        if self.include_gripper:
            losses["fm_grip"] = self.loss_fun(pred_vel[..., 6:7], target_vel[..., 6:7])
        else:
            losses["fm_grip"] = pos.new_zeros(())
        debug_tensors = {
            "y0": y0,
            "y1": y1,
            "ys": ys,
            "target_vel": target_vel,
            "flow_input": flow_input,
            "pred_vel": pred_vel,
            "flow_time": s,
            "xi_norm": xi_norm,
        }
        return losses, debug_tensors

    def _weighted(self, losses: dict[str, torch.Tensor], key: str, default: float) -> torch.Tensor:
        return losses[key] * self.l_w.get(key, default)

    def _tube_loss_total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self._weighted(losses, "center_xyz", 10.0)
            + self._weighted(losses, "center_so3", 10.0)
            + self._weighted(losses, "center_grip", 1.0)
            + self._weighted(losses, "chart", 10.0)
            + self._weighted(losses, "smooth_xyz", 0.1)
            + self._weighted(losses, "smooth_so3", 0.1)
        )

    def _flow_loss_total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self._weighted(losses, "fm_xyz", 10.0)
            + self._weighted(losses, "fm_so3", 10.0)
            + self._weighted(losses, "fm_grip", 1.0)
            + self._weighted(losses, "chart_fm", 1.0)
        )

    def _log_losses(
        self,
        prefix: str,
        losses: dict[str, torch.Tensor],
        total: torch.Tensor,
    ) -> None:
        metrics = {f"loss/{prefix}/{key}": value.item() for key, value in losses.items()}
        metrics[f"loss/{prefix}/total"] = total.item()
        self.logger.log_metrics(metrics)

    # ############### Training ################

    def forward(self, batch):
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch
        losses = self.calculate_loss(pcd, robot_state_obs, robot_state_pred)

        total = losses.pop("total")
        self._log_losses("train", losses, total)
        return total

    def calculate_loss(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        robot_state_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = self.obs_encoder(pcd, robot_state_obs)
        pos, rot, gripper = self._pfp_to_pose(robot_state_pred)

        detach_tube = self.training_stage == "flow" and self.freeze_tube_predictor
        tube = self._predict_tube(cond, detach_tube=detach_tube)
        losses = self._tube_losses(pos, rot, gripper, tube)
        debug_tensors = {
            "cond": cond,
            "target_pos": pos,
            "target_rot": rot,
            "target_gripper": gripper,
            "mu_p": tube["mu_p"],
            "mu_R": tube["mu_R"],
            "mu_R_6d": tube["mu_R_6d"],
        }
        if "mu_g" in tube:
            debug_tensors["mu_g"] = tube["mu_g"]

        if self.training_stage in ["flow", "joint"]:
            flow_losses, flow_debug_tensors = self._flow_losses(cond, pos, rot, gripper, tube)
            losses.update(flow_losses)
            debug_tensors.update(flow_debug_tensors)
            total = self._flow_loss_total(losses)
            if self.training_stage == "joint":
                total = total + self._tube_loss_total(losses)
        else:
            total = self._tube_loss_total(losses)

        losses["total"] = total
        debug_tensors.update({f"loss_{key}": value for key, value in losses.items()})
        force_debug = not torch.isfinite(total).all().item()
        self._debug_policy_stats("calculate_loss", debug_tensors, force=force_debug)
        self._debug_step += 1
        return losses

    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        batch = self._norm_data(batch)
        pcd, robot_state_obs, robot_state_pred = batch

        losses = self.calculate_loss(pcd, robot_state_obs, robot_state_pred)
        total = losses.pop("total")
        self._log_losses("eval", losses, total)

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
        noise: torch.Tensor | None = None,
        return_traj: bool = False,
    ) -> torch.Tensor:
        cond = self.obs_encoder(pcd, robot_state_obs)
        batch_size = cond.shape[0]
        device = cond.device
        tube = self._predict_tube(cond)

        if noise is not None:
            residual = noise
        elif self.inference_init == "smooth_noise":
            residual = self._sample_base_residual(batch_size, device)
        else:
            residual = self._zero_base_residual(batch_size, device)

        traj = [self._decode_action(residual, tube, robot_state_obs)]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        t0 = t0.to(device)
        dt = dt.to(device)
        for i in range(self.num_k_infer):
            timesteps = torch.full((batch_size,), t0[i], dtype=residual.dtype, device=device)
            model_time = timesteps * self.pos_emb_scale if self.time_conditioning else None
            flow_input = torch.cat([residual, self._mu_feature(tube)], dim=-1)
            vel = self.diffusion_net(flow_input, model_time, global_cond=cond)
            residual = residual.detach().clone() + vel * dt[i]
            residual[..., 3:6] = clamp_norm(residual[..., 3:6], self.tube_radius)
            traj.append(self._decode_action(residual, tube, robot_state_obs))

        return torch.stack(traj) if return_traj else traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int | None,
        flow_schedule: str | None = None,
        exp_scale: float | None = None,
        subs_factor: int = 1,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        cfg.model.subs_factor = subs_factor
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: TubeLocalFlowPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class TubeLocalFlowPolicyImage(TubeLocalFlowPolicy):
    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        return image.float() / 255.0

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
