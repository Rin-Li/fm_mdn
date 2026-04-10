import time
import numpy as np
import open3d as o3d
import spatialmath.base as sm
from pyrep.const import RenderMode
from pfp.envs.base_env import BaseEnv
from pyrep.errors import IKError
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.utils import name_to_task_class
from pfp.common.visualization import RerunViewer as RV
from pfp.common.o3d_utils import make_pcd, merge_pcds
from pfp.common.se3_utils import rot6d_to_quat_np, pfp_to_pose_np

try:
    import rerun as rr
except ImportError:
    print("WARNING: Rerun not installed. Visualization will not work.")


class RLBenchEnv(BaseEnv):
    """
    DT = 0.05 (50ms/20Hz)
    robot_state = [px, py, pz, r00, r10, r20, r01, r11, r21, gripper]
    The pose is the ttip frame, with x pointing backwards, y pointing left, and z pointing down.
    """

    def __init__(
        self,
        task_name: str,
        voxel_size: float,
        n_points: int,
        use_pc_color: bool,
        headless: bool,
        vis: bool,
        obs_mode: str = "pcd",
    ):
        assert obs_mode in ["pcd", "rgb"], "Invalid obs_mode"
        self.obs_mode = obs_mode
        # image_size=(128, 128)
        self.voxel_size = voxel_size
        self.n_points = n_points
        self.use_pc_color = use_pc_color
        camera_config = CameraConfig(
            rgb=True,
            depth=False,
            mask=False,
            point_cloud=True,
            image_size=(128, 128),
            render_mode=RenderMode.OPENGL,
        )
        obs_config = ObservationConfig(
            left_shoulder_camera=camera_config,
            right_shoulder_camera=camera_config,
            overhead_camera=camera_config,
            wrist_camera=camera_config,
            front_camera=camera_config,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )
        # EE pose is (X,Y,Z,Qx,Qy,Qz,Qw)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()
        )
        self.env = Environment(
            action_mode,
            obs_config=obs_config,
            headless=headless,
        )
        self.env.launch()
        self.task = self.env.get_task(name_to_task_class(task_name))
        self.robot_position = self.env._robot.arm.get_position()
        self.ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(self.robot_position[0] + 0.1, -0.65, self.robot_position[2] - 0.05),
            max_bound=(1, 0.65, 2),
        )
        self.vis = vis
        self.last_obs = None
        if self.vis:
            RV.add_axis("vis/origin", np.eye(4), size=0.01, timeless=True)
            RV.add_aabb(
                "vis/ws_aabb", self.ws_aabb.get_center(), self.ws_aabb.get_extent(), timeless=True
            )
        return

    def reset(self):
        self.task.reset()
        return

    def reset_rng(self):
        return

    def step(self, robot_state: np.ndarray):
        ee_position = robot_state[:3]
        ee_quat = rot6d_to_quat_np(robot_state[3:9])
        gripper = robot_state[-1:]
        action = np.concatenate([ee_position, ee_quat, gripper])
        reward, terminate = self._step_safe(action)
        return reward, terminate

    def _step_safe(self, action: np.ndarray, recursion_depth=0):
        if recursion_depth > 15:
            print("Warning: Recursion depth limit reached.")
            return 0.0, True
        try:
            _, reward, terminate = self.task.step(action)
        except IKError and InvalidActionError as e:
            print(e)
            cur_position = self.last_obs.gripper_pose[:3]
            des_position = action[:3]
            new_position = cur_position + (des_position - cur_position) * 0.25

            cur_quat = self.last_obs.gripper_pose[3:]
            cur_quat = np.array([cur_quat[3], cur_quat[0], cur_quat[1], cur_quat[2]])
            des_quat = action[3:7]
            des_quat = np.array([des_quat[3], des_quat[0], des_quat[1], des_quat[2]])
            new_quat = sm.qslerp(cur_quat, des_quat, 0.25, shortest=True)
            new_quat = np.array([new_quat[1], new_quat[2], new_quat[3], new_quat[0]])

            new_action = np.concatenate([new_position, new_quat, action[-1:]])
            reward, terminate = self._step_safe(new_action, recursion_depth + 1)
        return reward, terminate

    def get_obs(self) -> tuple[np.ndarray, ...]:
        obs_rlbench = self.task.get_observation()
        self.last_obs = obs_rlbench
        robot_state = self.get_robot_state(obs_rlbench)
        if self.obs_mode == "pcd":
            pcd_o3d = self.get_pcd(obs_rlbench)
            pcd = np.asarray(pcd_o3d.points)
            if self.use_pc_color:
                pcd_color = np.asarray(pcd_o3d.colors, dtype=np.float32)
                pcd = np.concatenate([pcd, pcd_color], axis=-1)
            obs = pcd
        elif self.obs_mode == "rgb":
            obs = self.get_images(obs_rlbench)
        return robot_state, obs

    def get_robot_state(self, obs: Observation) -> np.ndarray:
        ee_position = obs.gripper_matrix[:3, 3]
        ee_rot6d = obs.gripper_matrix[:3, :2].flatten(order="F")
        gripper = np.array([obs.gripper_open])
        robot_state = np.concatenate([ee_position, ee_rot6d, gripper])
        return robot_state

    def get_pcd(self, obs: Observation) -> o3d.geometry.PointCloud:
        right_pcd = make_pcd(obs.right_shoulder_point_cloud, obs.right_shoulder_rgb)
        left_pcd = make_pcd(obs.left_shoulder_point_cloud, obs.left_shoulder_rgb)
        overhead_pcd = make_pcd(obs.overhead_point_cloud, obs.overhead_rgb)
        front_pcd = make_pcd(obs.front_point_cloud, obs.front_rgb)
        wrist_pcd = make_pcd(obs.wrist_point_cloud, obs.wrist_rgb)
        pcd_list = [right_pcd, left_pcd, overhead_pcd, front_pcd, wrist_pcd]
        pcd = merge_pcds(self.voxel_size, self.n_points, pcd_list, self.ws_aabb)
        return pcd

    def get_images(self, obs: Observation) -> np.ndarray:
        images = np.stack(
            (
                obs.right_shoulder_rgb,
                obs.left_shoulder_rgb,
                obs.overhead_rgb,
                obs.front_rgb,
                obs.wrist_rgb,
            )
        )
        return images

    def vis_step(self, robot_state: np.ndarray, obs: np.ndarray, prediction: np.ndarray = None):
        """
        robot_state: the current robot state (10,)
        obs: either pcd or images
            - pcd: the current point cloud (N, 6) or (N, 3)
            - images: the current images (5, H, W, 3)
        prediction: the full trajectory of robot states (T, 10)
        """
        VIS_FLOW = False
        if not self.vis:
            return
        rr.set_time_seconds("time", time.time())

        # Point cloud
        if self.obs_mode == "pcd":
            pcd = obs
            pcd_xyz = pcd[:, :3]
            pcd_color = (pcd[:, 3:6] * 255).astype(np.uint8) if self.use_pc_color else None
            RV.add_np_pointcloud("vis/pcd_obs", points=pcd_xyz, colors_uint8=pcd_color, radii=0.003)

        # RGB images
        elif self.obs_mode == "rgb":
            images = obs
            for i, img in enumerate(images):
                RV.add_rgb(f"vis/rgb_obs_{i}", img)

        # EE State
        ee_pose = pfp_to_pose_np(robot_state[np.newaxis, ...]).squeeze()
        RV.add_axis("vis/ee_state", ee_pose)
        rr.log("plot/gripper_state", rr.Scalar(robot_state[-1]))

        if prediction is None:
            return

        # EE predictions
        final_pred = prediction[-1]
        if VIS_FLOW:
            for traj in prediction:
                RV.add_traj("vis/traj_k", traj)
        else:
            RV.add_traj("vis/ee_pred", final_pred)

        # Gripper action prediction
        rr.log("plot/gripper_pred", rr.Scalar(final_pred[0, -1]))
        return

    def close(self):
        self.env.shutdown()
        return


if __name__ == "__main__":
    env = RLBenchEnv(
        "close_microwave",
        voxel_size=0.01,
        n_points=5500,
        use_pc_color=False,
        headless=True,
        vis=True,
    )
    env.reset()
    for i in range(1000):
        robot_state, pcd = env.get_obs()
        next_robot_state = robot_state.copy()
        next_robot_state[:3] += np.array([-0.005, 0.005, 0.0])
        env.step(next_robot_state)
    env.close()
