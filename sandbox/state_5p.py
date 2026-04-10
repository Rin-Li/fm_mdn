import torch
import pypose as pp
from pfp.common.visualization import RerunViewer as RV
from pfp.common.se3_utils import pfp_to_state5p_th, state5p_to_pfp_th, pfp_to_pose_th

B = 2
T = 2
grippers = torch.ones(B, T, 1)
poses = pp.randn_SE3(B, T).matrix()
state_pfp = torch.cat([poses[..., :3, 3], poses[..., :3, 0], poses[..., :3, 1], grippers], dim=-1)
state_5p = pfp_to_state5p_th(state_pfp)
state_5p_pcds = state_5p[:, :, :15].reshape(B, T, 5, 3)

noise = torch.randn_like(state_5p) * 0.01
noise[..., 15] = 0
noisy_state_5p = state_5p + noise
noisy_5p_pcds = noisy_state_5p[:, :, :15].reshape(B, T, 5, 3)
estimated_state_pfp = state5p_to_pfp_th(noisy_state_5p)
estimated_poses, gripper = pfp_to_pose_th(estimated_state_pfp)

RV("state_5p")

for i in range(B):
    for j in range(T):
        RV.add_axis(f"state_pfp_{i}_{j}", poses[i, j].numpy())
        RV.add_axis(f"estimated_{i}_{j}", estimated_poses[i, j].numpy())
        RV.add_np_pointcloud(f"state_5p_{i}_{j}", state_5p_pcds[i, j].numpy(), radii=0.005)
        RV.add_np_pointcloud(f"noisy_5p_{i}_{j}", noisy_5p_pcds[i, j].numpy(), radii=0.005)


print("done")
