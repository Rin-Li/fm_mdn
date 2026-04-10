from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    The base abstract class for all envs.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_rng(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_obs(self):
        pass
