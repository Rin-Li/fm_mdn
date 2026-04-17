from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import rot6d_to_rotmat_th


def _so3_relative_angle_to_components(
    pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor
) -> torch.Tensor:
    """Pairwise geodesic angle between K predicted components and a target rotation."""
    pred_rotmat = rot6d_to_rotmat_th(pred_rot6d)  # (B, T, K, 3, 3)
    target_rotmat = rot6d_to_rotmat_th(target_rot6d).unsqueeze(2)  # (B, T, 1, 3, 3)
    rel = pred_rotmat.transpose(-1, -2) @ target_rotmat
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    return torch.acos(cos_theta)


def so3_mixture_nll_th(
    alpha_logits: torch.Tensor,
    mu_rot6d: torch.Tensor,
    sigma_raw: torch.Tensor,
    target_rot6d: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate negative log likelihood on SO(3) with isotropic Gaussian components
    defined in the tangent space around each component mean.
    """
    sigma = F.softplus(sigma_raw) + 1e-4
    angle = _so3_relative_angle_to_components(mu_rot6d, target_rot6d)
    log_alpha = F.log_softmax(alpha_logits, dim=-1)
    # Approximate isotropic Gaussian in so(3), omitting constants independent of parameters.
    log_component = log_alpha - 0.5 * (angle ** 2) / (sigma ** 2) - 3.0 * torch.log(sigma)
    return -torch.logsumexp(log_component, dim=-1).mean()


class SO3AutoregressiveMixtureHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int = 512,
        q_mix: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_mix = q_mix
        self.init_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(6, hidden_dim)
        self.head_alpha = nn.Linear(hidden_dim, q_mix)
        self.head_mu = nn.Linear(hidden_dim, q_mix * 6)
        self.head_sigma = nn.Linear(hidden_dim, q_mix)

    def step(
        self,
        prev_rot6d: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.gru(prev_rot6d, hidden)
        alpha_logits = self.head_alpha(hidden)
        mu_rot6d = self.head_mu(hidden).view(hidden.shape[0], self.q_mix, 6)
        sigma_raw = self.head_sigma(hidden)
        return alpha_logits, mu_rot6d, sigma_raw, hidden

    def init_hidden(self, cond: torch.Tensor) -> torch.Tensor:
        return self.init_net(cond)


class SO3PriorPolicy(ComposerModel):
    def __init__(
        self,
        x_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        obs_encoder: nn.Module,
        prior_head: nn.Module,
        norm_pcd_center: list,
        augment_data: bool = False,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.obs_encoder = obs_encoder
        self.prior_head = prior_head
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, robot_state_pred = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)
        robot_state_pred = self._norm_robot_state(robot_state_pred)
        return pcd, robot_state_obs, robot_state_pred

    def _get_start_rot6d(self, robot_state_obs: torch.Tensor) -> torch.Tensor:
        return robot_state_obs[:, -1, 3:9]

    def _teacher_forcing_inputs(
        self, robot_state_obs: torch.Tensor, target_rot6d: torch.Tensor
    ) -> torch.Tensor:
        start = self._get_start_rot6d(robot_state_obs).unsqueeze(1)
        return torch.cat([start, target_rot6d[:, :-1]], dim=1)

    def _rollout_teacher(
        self,
        cond: torch.Tensor,
        prev_rot6d_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = prev_rot6d_seq.shape
        hidden = self.prior_head.init_hidden(cond)
        alpha_logits_list, mu_rot6d_list, sigma_raw_list = [], [], []
        for t in range(T):
            alpha_logits, mu_rot6d, sigma_raw, hidden = self.prior_head.step(
                prev_rot6d_seq[:, t], hidden
            )
            alpha_logits_list.append(alpha_logits)
            mu_rot6d_list.append(mu_rot6d)
            sigma_raw_list.append(sigma_raw)
        return (
            torch.stack(alpha_logits_list, dim=1),
            torch.stack(mu_rot6d_list, dim=1),
            torch.stack(sigma_raw_list, dim=1),
        )

    def forward(self, batch):
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        pcd, robot_state_obs, robot_state_pred = self._norm_data(batch)
        cond = self.obs_encoder(pcd, robot_state_obs)
        target_rot6d = robot_state_pred[..., 3:9]
        prev_rot6d_seq = self._teacher_forcing_inputs(robot_state_obs, target_rot6d)
        alpha_logits, mu_rot6d, sigma_raw = self._rollout_teacher(cond, prev_rot6d_seq)
        loss = so3_mixture_nll_th(alpha_logits, mu_rot6d, sigma_raw, target_rot6d)
        self.logger.log_metrics({"loss/train/so3_mixture_nll": loss.item()})
        return loss

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        pcd, robot_state_obs, robot_state_pred = self._norm_data(batch)
        cond = self.obs_encoder(pcd, robot_state_obs)
        target_rot6d = robot_state_pred[..., 3:9]
        prev_rot6d_seq = self._teacher_forcing_inputs(robot_state_obs, target_rot6d)
        alpha_logits, mu_rot6d, sigma_raw = self._rollout_teacher(cond, prev_rot6d_seq)
        loss = so3_mixture_nll_th(alpha_logits, mu_rot6d, sigma_raw, target_rot6d)
        pred_rot6d = self.infer_rot6d(pcd, robot_state_obs)
        mse_rot6d = nn.functional.mse_loss(pred_rot6d, target_rot6d)
        self.logger.log_metrics(
            {
                "loss/eval/so3_mixture_nll": loss.item(),
                "metrics/eval/mse_rot6d": mse_rot6d.item(),
            }
        )
        return pred_rot6d

    @torch.no_grad()
    def infer_rot6d(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        normalized_inputs: bool = False,
    ) -> torch.Tensor:
        if not normalized_inputs:
            pcd = self._norm_obs(pcd.clone())
            robot_state_obs = self._norm_robot_state(robot_state_obs.clone())
        cond = self.obs_encoder(pcd, robot_state_obs)
        hidden = self.prior_head.init_hidden(cond)
        prev_rot6d = self._get_start_rot6d(robot_state_obs)
        samples = []
        for _ in range(self.n_pred_steps):
            alpha_logits, mu_rot6d, sigma_raw, hidden = self.prior_head.step(prev_rot6d, hidden)
            alpha = F.softmax(alpha_logits, dim=-1)
            sigma = F.softplus(sigma_raw) + 1e-4
            cat = torch.distributions.Categorical(alpha)
            idx = cat.sample()
            batch_idx = torch.arange(prev_rot6d.shape[0], device=prev_rot6d.device)
            mu_sel = mu_rot6d[batch_idx, idx]
            sigma_sel = sigma[batch_idx, idx].unsqueeze(-1)

            # Sample in the tangent space around the component mean, then map back to SO(3).
            mu_rotmat = rot6d_to_rotmat_th(mu_sel)
            mu_so3 = pp.mat2SO3(mu_rotmat, check=False)
            tangent_noise = pp.so3(torch.randn((*mu_sel.shape[:-1], 3), device=mu_sel.device))
            sample_so3 = mu_so3 @ pp.Exp(tangent_noise * sigma_sel)
            sample_rotmat = pp.matrix(sample_so3)
            sample_rot6d = sample_rotmat[..., :3, :2].mT.flatten(start_dim=-2)
            samples.append(sample_rot6d)
            prev_rot6d = sample_rot6d
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def infer_so3(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        normalized_inputs: bool = False,
    ) -> pp.SO3:
        pred_rot6d = self.infer_rot6d(pcd, robot_state_obs, normalized_inputs=normalized_inputs)
        pred_rotmat = rot6d_to_rotmat_th(pred_rot6d)
        return pp.mat2SO3(pred_rotmat, check=False)

    @classmethod
    def load_from_checkpoint(cls, ckpt_name: str, ckpt_episode: str):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: SO3PriorPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        return model
