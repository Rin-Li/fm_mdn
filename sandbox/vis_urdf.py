import time
import numpy as np
import rerun as rr
from pfp import REPO_DIRS
from pfp.common.visualization import RerunViewer as RV
from pfp.common.visualization import RerunURDF

panda_dir = REPO_DIRS.URDFS / "panda"
meshes_root = panda_dir
rerun_panda = RerunURDF("vis/panda", panda_dir / "panda.urdf", meshes_root)
rerun_panda_gripper = RerunURDF("vis/panda_gripper", panda_dir / "panda_gripper.urdf", meshes_root)

print("Actuated Joints: ", [j.name for j in rerun_panda.urdf.actuated_joints])
print("Current joint state: ", rerun_panda.urdf.cfg)


RV()
RV.add_axis("vis/origin", np.eye(4), size=0.01, timeless=True)
root_pose = np.eye(4)
root_pose_gripper = np.eye(4)
for i in range(10):
    rr.set_time_seconds("timestep", time.time())
    # Panda
    joint_state = np.ones(7) * i / 10
    joint_state = np.concatenate([joint_state, [0.04, 0.04]])
    rerun_panda.update_vis(joint_state, root_pose)

    # Panda Gripper
    joint_state_gripper = np.array([0.04, 0.04])
    root_pose_gripper[:3, 3] = i / 10
    rerun_panda_gripper.update_vis(joint_state_gripper, root_pose_gripper)


print("done")
