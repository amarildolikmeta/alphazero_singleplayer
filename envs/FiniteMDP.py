import numpy as np
from mushroom.environments.environment import  MDPInfo
from mushroom.utils import spaces
from gym.utils import seeding
from gym import spaces as gym_spaces


class FiniteMDP:
    """
    Finite Markov Decision Process.

    """
    def __init__(self, p, rew, mu=None, gamma=.9, horizon=np.inf):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            mu (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon.

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        observation_space = spaces.Discrete(p.shape[0])
        action_space = spaces.Discrete(p.shape[1])

        self.observation_space = gym_spaces.Discrete(p.shape[0])
        self.action_space = gym_spaces.Discrete(p.shape[1])
        horizon = horizon
        gamma = gamma
        self._mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        self.seed()
        self.t = 0
        #super().__init__(mdp_info)

    def get_signature(self):
        sig = {'state': np.copy(self._state)}
        return sig

    def set_signature(self, sig):
        self._state = np.copy(sig['state'])

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [self.np_random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([self.np_random.choice(self.p.shape[0])])
        else:
            self._state = state
        self.t = 0
        return self._state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if np.isscalar(action):
            action = [action]
        p = self.p[self._state[0], action[0], :]
        if np.sum(p) != 1:
            print("ASD")
        next_state = np.array([self.np_random.choice(p.size, p=p)])

        absorbing = not np.any(self.p[next_state[0], :, :])
        reward = self.r[self._state[0], action[0], next_state[0]]
        self.t += 1
        self._state = next_state

        return self._state, reward, absorbing, {}

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info
