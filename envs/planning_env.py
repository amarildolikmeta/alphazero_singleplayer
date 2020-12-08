from gym import Env
import numpy as np
from gym.utils import seeding


class PlanningEnv(Env):

    def __init__(self):
        self.np_random = None
        super(PlanningEnv, self).__init__()
        self.action_space = None

    def get_max_episode_length(self) -> int:
        return np.inf

    def get_distance_to_horizon(self) -> int:
        return np.inf

    def is_terminal(self) -> bool:
        raise NotImplementedError("is_terminal method must be implemented by subclasses of PlanningEnv")

    def get_mean_estimation(self, action, owner) -> float:
        raise NotImplementedError("get_mean_estimation method must be implemented by subclasses of PlanningEnv")

    def get_available_actions(self, agent: int) -> list:
        assert self.action_space is not None, "Action space has not been initialized"
        return [i for i in range(self.action_space.n)]

    def get_next_agent(self) -> int:
        return 0

    def step(self, action):
        raise NotImplementedError("step method must be implemented by subclasses of PlanningEnv")

    def reset(self):
        raise NotImplementedError("reset method must be implemented by subclasses of PlanningEnv")

    def enable_search_mode(self) -> None:
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print("[WARNING] No render mode is implemented for this environment")

    def save_results(self, timestamp):
        print("[WARNING] No save method is implemented for this environment")

    def reset_stochasticity(self):
        pass

    def get_max_ep_length(self):
        return np.inf
