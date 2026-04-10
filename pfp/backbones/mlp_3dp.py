import torch
import torch.nn as nn
from diffusion_policy_3d.model.vision.pointnet_extractor import (
    PointNetEncoderXYZRGB,
    PointNetEncoderXYZ,
)


class MLP3DP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels == 3:
            self.backbone = PointNetEncoderXYZ(
                in_channels=in_channels,
                out_channels=out_channels,
                use_layernorm=True,
                final_norm="layernorm",
                normal_channel=False,
            )
        elif in_channels == 6:
            self.backbone = PointNetEncoderXYZRGB(
                in_channels=in_channels,
                out_channels=out_channels,
                use_layernorm=True,
                final_norm="layernorm",
                normal_channel=False,
            )
        else:
            raise ValueError("Invalid number of input channels for MLP3DP")
        return

    def forward(self, pcd: torch.Tensor, robot_state_obs: torch.Tensor = None) -> torch.Tensor:
        B = pcd.shape[0]
        # Flatten the batch and time dimensions
        pcd = pcd.float().reshape(-1, *pcd.shape[2:])
        robot_state_obs = robot_state_obs.float().reshape(-1, *robot_state_obs.shape[2:])
        # Encode all point clouds (across time steps and batch size)
        encoded_pcd = self.backbone(pcd)
        nx = torch.cat([encoded_pcd, robot_state_obs], dim=1)
        # Reshape back to the batch dimension. Now the features of each time step are concatenated
        nx = nx.reshape(B, -1)
        return nx
