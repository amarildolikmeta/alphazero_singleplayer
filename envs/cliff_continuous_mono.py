import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym import register
from copy import copy


def generate_cliff(**kwargs):
    if kwargs is None:
        kwargs = {}
    return CliffWorldContinuousMono(**kwargs)


class CliffWorldContinuousMono(gym.Env):
    def __init__(self, dim=(7,), gamma=0.99, small=-1, large=-5, max_action=1, sigma_noise=0.1, horizon=np.inf):
        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim
        self.min_action = np.array([-max_action])
        self.max_action = np.array([max_action])
        self.min_position = np.array([0])
        self.max_position = np.array([dim[0]])

        self.starting_state_center = np.array([0.5])
        self.goal_state_center = np.array([dim[0] - 0.5])

        self.sigma_noise = sigma_noise
        self.viewer = None
        self.cliff = np.array([3., 3.5])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_goal(self, new_state):
        return np.max(np.abs(new_state - self.goal_state_center)) <= 0.5

    def _is_cliff(self, new_state):
        return self.cliff[0] <= new_state[0] <= self.cliff[1]

    def _generate_initial_state(self):
        retry = True
        while retry:
            new_state = self.starting_state_center + self.sigma_noise * self.np_random.randn(1)
            new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
            retry = self._is_cliff(new_state)
        return new_state

    def action_traslation(self, a):
        if a == 0:
            action = np.array(0.)  # do not move
        elif a == 1:
            action = np.array(1.)  # move forward
        elif a == 2:
            action = np.array(-1.)  # move backward
        return action

    def step(self, action):
        if self._t >= self.horizon or self.done:
            return self.state, 0, True, {}

        is_goal = self._is_goal(self.state)
        if is_goal or self._t >= self.horizon:
            return self.state, 0, True, {}
        action = self.action_traslation(action)

        new_state = self.state + action + self.np_random.randn(1) * self.sigma_noise
        new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)

        is_cliff = self._is_cliff(new_state)
        is_goal = self._is_goal(new_state)

        reward = self.small
        self._t += 1
        done = False

        if is_cliff:
            new_state = self._generate_initial_state()
            reward = self.large

        if is_goal:
            done = True

        if self._t >= self.horizon:
            done = True
        self.state = new_state
        reward /= abs(self.large)
        #reward += 1
        self.done = done
        return self.state, reward, done, {}

    def reset(self):
        self._t = 0
        self.state = self._generate_initial_state()
        self.done = False
        return self.state

    def get_signature(self):
        sig = {'agent_pos': np.copy(self.state), 't': copy(self._t), 'done': self.done}
        return sig

    def set_signature(self, sig):
        self.state = np.copy(sig['agent_pos'])
        self._t = copy(sig['t'])
        self.done = sig['done']

register(
    id='Cliff-v0',
    entry_point='envs.cliff_continuous_mono:CliffWorldContinuousMono'
)

if __name__ == '__main__':
    mdp = CliffWorldContinuousMono(sigma_noise=0., horizon=20)

    s = mdp.reset()
    rets = []
    timesteps = 5000
    count = 0
    n = 1000
    for i in range(n):
        t = 0
        ret = 0
        s = mdp.reset()
        while t < 100:
            print("State: ", s)
            a = 1
            s, r, done, _ = mdp.step(a)
            count += 1
            print("Reward: ", r)
            ret += r
            t += 1
            if done:

                break
            if count > timesteps:
                break
        if count <= timesteps:
            print("Return:", ret)
            rets.append(ret)
        else:
            break
    print("Average Return:", np.mean(rets))
    print("Average error:", np.std(rets) / np.sqrt(len(rets)))
    print("Nume episodes:", len(rets))