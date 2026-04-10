from __future__ import annotations
import torch
import numpy as np
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp import DATA_DIRS


class RobotDatasetImages(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        subs_factor: int = 1,  # 1 means no subsampling
        **kwargs,
    ) -> None:
        """
        To me it makes sense that sequence_length == n_obs_steps + n_prediction_steps
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "images"]
        data_key_first_k = {"images": n_obs_steps * subs_factor}
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        images = sample["images"][: cur_step_i : self.subs_factor]
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor]
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor]
        return images, robot_state_obs, robot_state_pred


if __name__ == "__main__":
    dataset = RobotDatasetImages(
        data_path=DATA_DIRS.PFP / "open_fridge" / "train",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
    )
    i = 20
    obs, robot_state_obs, robot_state_pred = dataset[i]
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")
