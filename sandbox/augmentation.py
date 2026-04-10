import copy
import numpy as np
from torch.utils.data import DataLoader
from pfp import DATA_DIRS
from pfp.data.dataset_pcd import RobotDatasetPcd, augment_pcd_data
from pfp.common.visualization import RerunViewer as RV
from pfp.common.visualization import RerunTraj
import rerun as rr

rr_traj = {
    "original_robot_obs": RerunTraj(),
    "augmented_robot_obs": RerunTraj(),
    "original_prediction": RerunTraj(),
    "augmented_prediction": RerunTraj(),
}


def vis_batch(name, batch):
    pcd, robot_state_obs, robot_state_pred = batch
    pcd = pcd[0, -1].cpu().numpy()
    robot_state_obs = robot_state_obs[0].cpu().numpy()
    robot_state_pred = robot_state_pred[0].cpu().numpy()
    RV.add_np_pointcloud(
        f"vis/{name}_pcd", points=pcd[:, :3], colors_uint8=(pcd[:, 3:6] * 255).astype(np.uint8)
    )
    rr_traj[f"{name}_robot_obs"].add_traj(f"{name}_robot_obs", robot_state_obs, size=0.008)
    rr_traj[f"{name}_prediction"].add_traj(f"{name}_prediction", robot_state_pred)
    return


RV("augmentation_vis")
RV.add_axis("vis/origin", np.eye(4), timeless=True)

task_name = "sponge_on_plate"

data_path_train = DATA_DIRS.PFP_REAL / task_name / "train"
dataset_train = RobotDatasetPcd(
    data_path_train,
    n_obs_steps=2,
    n_pred_steps=32,
    subs_factor=3,
    use_pc_color=False,
    n_points=4096,
)
dataloader_train = DataLoader(
    dataset_train,
    shuffle=False,
    batch_size=1,
    persistent_workers=False,
)

for i, batch in enumerate(dataloader_train):
    rr.set_time_sequence("step", i)
    original_batch = copy.deepcopy(batch)
    vis_batch("original", original_batch)

    augmented_batch = copy.deepcopy(batch)
    augmented_batch = augment_pcd_data(augmented_batch)
    vis_batch("augmented", augmented_batch)

    if i > 500:
        break
