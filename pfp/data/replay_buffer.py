from __future__ import annotations
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codec, Jpeg2k

register_codec(Jpeg2k)


class RobotReplayBuffer(ReplayBuffer):
    def __init__(self, root: zarr.Group):
        super().__init__(root)
        self.jpeg_compressor = Jpeg2k()
        return

    def add_episode_from_list(self, data_list: list[dict[str, np.ndarray]], **kwargs):
        """
        data_list is a list of dictionaries, where each dictionary contains the data for one step.
        """
        data_dict = dict()
        for key in data_list[0].keys():
            data_dict[key] = np.stack([x[key] for x in data_list])
        self.add_episode(data_dict, **kwargs)
        return

    def add_episode_from_list_compressed(self, data_list: list[dict[str, np.ndarray]], **kwargs):
        """
        data_list is a list of dictionaries, where each dictionary contains the data for one step.
        WARNING: decoding (i.e. reading) is broken.
        """
        data_dict = {key: np.stack([x[key] for x in data_list]) for key in data_list[0].keys()}
        # get the keys starting with 'rgb*'
        rgb_keys = [key for key in data_dict.keys() if key.startswith("rgb")]
        rgb_shapes = [data_list[0][key].shape for key in rgb_keys]
        chunks = {rgb_keys[i]: (1, *rgb_shapes[i]) for i in range(len(rgb_keys))}
        compressors = {key: self.jpeg_compressor for key in rgb_keys}
        self.add_episode(data_dict, chunks, compressors, **kwargs)
        return
