import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from rlbench.backend.observation import Observation
from pfp import DATA_DIRS, set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.visualization import RerunViewer as RV


# For valid, call it with: --config-name=collect_demos_valid
# To actually save the data, remember to call it with: save_data=True
@hydra.main(version_base=None, config_path="../conf", config_name="collect_demos_train")
def main(cfg: OmegaConf):
    set_seeds(cfg.seed)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    print(OmegaConf.to_yaml(cfg))

    assert cfg.mode in ["train", "valid"]
    if cfg.env_config.vis:
        RV("pfp_collect_demos")
    env = RLBenchEnv(use_pc_color=True, **cfg.env_config)
    if cfg.save_data:
        data_path = DATA_DIRS.PFP / cfg.env_config.task_name / cfg.mode
        if data_path.is_dir():
            print(f"ERROR: Data path {data_path} already exists! Exiting...")
            return
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="a")

    for _ in tqdm(range(cfg.num_episodes)):
        data_history = list()
        demo = env.task.get_demos(1, live_demos=True)[0]
        observations: list[Observation] = demo._observations
        for obs in observations:
            robot_state = env.get_robot_state(obs)
            images = env.get_images(obs)
            pcd = env.get_pcd(obs)
            pcd_xyz = np.asarray(pcd.points)
            pcd_color = np.asarray(pcd.colors)
            data_history.append(
                {
                    "pcd_xyz": pcd_xyz.astype(np.float32),
                    "pcd_color": (pcd_color * 255).astype(np.uint8),
                    "robot_state": robot_state.astype(np.float32),
                    "images": images,
                }
            )
            env.vis_step(robot_state, np.concatenate((pcd_xyz, pcd_color), axis=-1))

        if cfg.save_data:
            replay_buffer.add_episode_from_list(data_history, compressors="disk")
            print(f"Saved episode with {len(data_history)} steps to disk.")

        # while True:
        #     env.step(robot_state)
    return


if __name__ == "__main__":
    main()
