from __future__ import annotations

import torch
import torch.nn.functional as F


def hat(xi: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors (..., 3) to skew matrices (..., 3, 3)."""
    x, y, z = xi.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def vee(mat: torch.Tensor) -> torch.Tensor:
    """Convert skew matrices (..., 3, 3) to axis-angle vectors (..., 3)."""
    return torch.stack(
        [
            mat[..., 2, 1],
            mat[..., 0, 2],
            mat[..., 1, 0],
        ],
        dim=-1,
    )


def so3_exp(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from local SO(3) tangent vectors to rotation matrices."""
    theta2 = (xi * xi).sum(dim=-1)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    theta4 = theta2 * theta2

    small = theta < 1e-4
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)

    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta4 / 120.0,
        torch.sin(theta) / theta_safe,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta4 / 720.0,
        (1.0 - torch.cos(theta)) / theta2_safe,
    )

    k = hat(xi)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    eye = eye.expand(*xi.shape[:-1], 3, 3)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def _so3_log_near_pi(rot: torch.Tensor, theta: torch.Tensor, eps: float) -> torch.Tensor:
    r00 = rot[..., 0, 0]
    r11 = rot[..., 1, 1]
    r22 = rot[..., 2, 2]

    x0 = 0.5 * torch.sqrt((1.0 + r00 - r11 - r22).clamp_min(eps))
    y0 = (rot[..., 0, 1] + rot[..., 1, 0]) / (4.0 * x0)
    z0 = (rot[..., 0, 2] + rot[..., 2, 0]) / (4.0 * x0)
    axis0 = torch.stack([x0, y0, z0], dim=-1)

    y1 = 0.5 * torch.sqrt((1.0 - r00 + r11 - r22).clamp_min(eps))
    x1 = (rot[..., 0, 1] + rot[..., 1, 0]) / (4.0 * y1)
    z1 = (rot[..., 1, 2] + rot[..., 2, 1]) / (4.0 * y1)
    axis1 = torch.stack([x1, y1, z1], dim=-1)

    z2 = 0.5 * torch.sqrt((1.0 - r00 - r11 + r22).clamp_min(eps))
    x2 = (rot[..., 0, 2] + rot[..., 2, 0]) / (4.0 * z2)
    y2 = (rot[..., 1, 2] + rot[..., 2, 1]) / (4.0 * z2)
    axis2 = torch.stack([x2, y2, z2], dim=-1)

    case0 = (r00 >= r11) & (r00 >= r22)
    case1 = (~case0) & (r11 >= r22)
    axis = torch.where(case0[..., None], axis0, torch.where(case1[..., None], axis1, axis2))
    axis = F.normalize(axis, dim=-1, eps=eps)
    return axis * theta[..., None]


def so3_log(rot: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log map from rotation matrices (..., 3, 3) to axis-angle vectors."""
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    skew_vec = vee(rot - rot.transpose(-1, -2))
    small = theta < 1e-4
    near_pi = (torch.pi - theta) < 1e-4
    safe_sin = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)

    general = (theta / (2.0 * safe_sin))[..., None] * skew_vec
    small_angle = 0.5 * skew_vec
    near_pi_angle = _so3_log_near_pi(rot, theta, eps)

    out = torch.where(small[..., None], small_angle, general)
    return torch.where(near_pi[..., None], near_pi_angle, out)


def rotation_6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert this repository's 6D rotation convention (..., 6) to SO(3)."""
    rot6d = torch.nan_to_num(rot6d, nan=0.0, posinf=0.0, neginf=0.0)
    x_raw = rot6d[..., :3]
    y_raw = rot6d[..., 3:6]
    fallback_x = torch.zeros_like(x_raw)
    fallback_x[..., 0] = 1.0
    fallback_y = torch.zeros_like(y_raw)
    fallback_y[..., 1] = 1.0

    x_norm = x_raw.norm(dim=-1, keepdim=True)
    x = torch.where(x_norm > 1e-6, x_raw / x_norm.clamp_min(1e-6), fallback_x)

    y_orth = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    y_norm = y_orth.norm(dim=-1, keepdim=True)
    y = torch.where(y_norm > 1e-6, y_orth / y_norm.clamp_min(1e-6), fallback_y)
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1, eps=1e-6)

    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-6)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def matrix_to_rotation_6d(rot: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) matrices to this repository's column-major 6D convention."""
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def geodesic_distance(rot_a: torch.Tensor, rot_b: torch.Tensor) -> torch.Tensor:
    """Return SO(3) geodesic distances for matching leading dimensions."""
    rel = rot_a.transpose(-1, -2) @ rot_b
    return so3_log(rel).norm(dim=-1)


def encode_residual(
    p_demo: torch.Tensor,
    rot_demo: torch.Tensor,
    mu_p: torch.Tensor,
    mu_rot: torch.Tensor,
) -> torch.Tensor:
    """Encode poses into tube-local position and SO(3) tangent residuals."""
    delta_p = p_demo - mu_p
    xi_rot = so3_log(mu_rot.transpose(-1, -2) @ rot_demo)
    return torch.cat([delta_p, xi_rot], dim=-1)


def decode_residual(
    residual: torch.Tensor,
    mu_p: torch.Tensor,
    mu_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode tube-local position and SO(3) tangent residuals back to poses."""
    delta_p = residual[..., :3]
    xi_rot = residual[..., 3:6]
    p = mu_p + delta_p
    rot = mu_rot @ so3_exp(xi_rot)
    return p, rot
