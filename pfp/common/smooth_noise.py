from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_smooth_noise(
    batch_size: int,
    horizon: int,
    dim: int,
    n_ctrl: int = 8,
    sigma: float = 0.1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample temporally smooth noise by linearly interpolating control points."""
    ctrl = torch.randn(batch_size, n_ctrl, dim, device=device) * sigma
    ctrl = ctrl.permute(0, 2, 1)
    noise = F.interpolate(ctrl, size=horizon, mode="linear", align_corners=True)
    return noise.permute(0, 2, 1)


def clamp_norm(x: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    """Clamp vectors to a maximum norm along the last dimension."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def sample_residual_y0(
    batch_size: int,
    horizon: int,
    sigma_p: float = 0.03,
    sigma_r: float = 0.15,
    r0: float = 0.8 * math.pi / 3.0,
    n_ctrl: int = 8,
    device: torch.device | str | None = None,
    include_gripper: bool = False,
    sigma_g: float = 0.1,
) -> torch.Tensor:
    """Sample smooth local residuals for position, orientation, and optionally gripper."""
    delta_p0 = sample_smooth_noise(batch_size, horizon, 3, n_ctrl, sigma_p, device)
    xi_r0 = sample_smooth_noise(batch_size, horizon, 3, n_ctrl, sigma_r, device)
    xi_r0 = clamp_norm(xi_r0, r0)
    parts = [delta_p0, xi_r0]
    if include_gripper:
        delta_g0 = sample_smooth_noise(batch_size, horizon, 1, n_ctrl, sigma_g, device)
        parts.append(delta_g0)
    return torch.cat(parts, dim=-1)
