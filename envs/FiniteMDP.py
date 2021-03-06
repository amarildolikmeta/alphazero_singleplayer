from copy import copy
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces as gym_spaces


class MDPInfo:
    """
    This class is used to store the information of the environment.

    """
    def __init__(self, observation_space, action_space, gamma, horizon):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space;
             action_space ([Box, Discrete]): the action space;
             gamma (float): the discount factor;
             horizon (int): the horizon.

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only
            works for discrete spaces.

        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action
            spaces.

        """
        return self.observation_space.shape + self.action_space.shape

class FiniteMDP(gym.Env):
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

        # Time horizon for game termination
        self._t = 0
        self.horizon = horizon

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        observation_space = gym_spaces.Discrete(p.shape[0])
        action_space = gym_spaces.Discrete(p.shape[1])

        self.observation_space = observation_space
        self.action_space = action_space
        horizon = horizon
        gamma = gamma
        self._mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        self.seed()
        #super().__init__(mdp_info)

    def get_state(self):
        return self._state

    def get_signature(self):
        sig = {'state': np.copy(self._state), 't': copy(self._t)}
        return sig

    def set_signature(self, sig):
        self._state = np.copy(sig['state'])
        self._t = copy(sig['t'])

    def reset(self, state=None):
        self._t = 0

        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [self.np_random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([self.np_random.choice(self.p.shape[0])])
        else:
            self._state = state
        return self._state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Increment timestep
        if self._t >= self.horizon:
            return self._state, 0, True, {}


        if np.isscalar(action):
            action = [action]
        p = self.p[self._state[0], action[0], :]
        if np.sum(p) != 1:
            print("ASD")
        next_state = np.array([self.np_random.choice(p.size, p=p)])

        absorbing = not np.any(self.p[next_state[0], :, :])
        reward = self.r[self._state[0], action[0], next_state[0]]
        self._t += 1
        # Signal that the state is terminal, if the time horizon has been reached
        if self._t >= self.horizon:
            absorbing = True

        self._state = next_state

        return self._state, reward, absorbing, {}

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info
