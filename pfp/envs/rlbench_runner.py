import wandb
from tqdm import tqdm
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.policy.base_policy import BasePolicy


class RLBenchRunner:
    def __init__(
        self,
        num_episodes: int,
        max_episode_length: int,
        env_config: dict,
        verbose=False,
    ) -> None:
        self.env: RLBenchEnv = RLBenchEnv(**env_config)
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.verbose = verbose
        return

    def run(self, policy: BasePolicy):
        wandb.define_metric("success", summary="mean")
        wandb.define_metric("steps", summary="mean")
        success_list: list[bool] = []
        steps_list: list[int] = []
        self.env.reset_rng()
        for episode in tqdm(range(self.num_episodes)):
            policy.reset_obs()
            self.env.reset()
            for step in range(self.max_episode_length):
                robot_state, obs = self.env.get_obs()
                prediction = policy.predict_action(obs, robot_state)
                self.env.vis_step(robot_state, obs, prediction)
                next_robot_state = prediction[-1, 0]  # Last K step, first T step
                reward, terminate = self.env.step(next_robot_state)
                success = bool(reward)
                if success or terminate:
                    break
            success_list.append(success)
            if success:
                steps_list.append(step)
            if self.verbose:
                print(f"Steps: {step}")
                print(f"Success: {success}")
            wandb.log({"episode": episode, "success": int(success), "steps": step})
        return success_list, steps_list
