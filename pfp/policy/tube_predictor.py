from __future__ import annotations

import torch
import torch.nn as nn

from pfp.common.so3 import rotation_6d_to_matrix


class TubePredictor(nn.Module):
    """Predict a condition-dependent pose tube center over the action horizon."""

    def __init__(
        self,
        cond_dim: int,
        horizon: int,
        hidden_dim: int = 512,
        predict_gripper: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.predict_gripper = predict_gripper
        self.out_dim = 10 if predict_gripper else 9

        head = nn.Linear(hidden_dim, horizon * self.out_dim)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            head,
        )
        self._init_head(head)

    def _init_head(self, head: nn.Linear) -> None:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        with torch.no_grad():
            bias = head.bias.view(self.horizon, self.out_dim)
            identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=bias.device)
            bias[:, 3:9] = identity_6d

    def forward(self, cond: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = cond.shape[0]
        out = self.net(cond).view(batch_size, self.horizon, self.out_dim)
        mu_p = out[..., :3]
        mu_r_6d = out[..., 3:9]
        mu_r = rotation_6d_to_matrix(mu_r_6d)
        pred = {
            "mu_p": mu_p,
            "mu_R": mu_r,
            "mu_R_6d": mu_r_6d,
        }
        if self.predict_gripper:
            pred["mu_g"] = out[..., 9:10]
        return pred
