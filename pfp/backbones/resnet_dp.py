import torch
import torch.nn as nn
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class ResnetDP(nn.Module):
    def __init__(self, shape_meta: dict):
        super().__init__()
        rgb_model = get_resnet(name="resnet18")
        self.backbone = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=(76, 76),
            random_crop=True,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )
        return

    def forward(self, images: torch.Tensor, robot_state_obs: torch.Tensor = None) -> torch.Tensor:
        B = images.shape[0]
        # Flatten the batch and time dimensions
        images = images.reshape(-1, *images.shape[2:]).permute(0, 1, 4, 2, 3)
        robot_state_obs = robot_state_obs.float().reshape(-1, *robot_state_obs.shape[2:])
        # Encode all observations (across time steps and batch size)
        obs_dict = {f"img_{i}": images[:, i] for i in range(images.shape[1])}
        obs_dict["robot_state"] = robot_state_obs
        nx = self.backbone(obs_dict)
        # Reshape back to the batch dimension. Now the features of each time step are concatenated
        nx = nx.reshape(B, -1)
        return nx
