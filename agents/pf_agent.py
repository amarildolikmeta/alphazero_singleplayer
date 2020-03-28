from rl.make_game import make_game
import numpy as np

import multiprocessing

def random_rollout(actions, env):
    done = False
    while not done:
        action = np.random.choice(actions)
        s, r, done, _ = env.step(action)
        if done:
            return r


class State(object):
    def __init__(self, particles, na, envs):
        self.v = self.evaluate(envs)
        self.particles = particles
        self.n = 1
        self.na = na

    def update(self):
        self.n += 1

    def evaluate(self, envs):

        actions = np.arange(self.na, dtype=int)

        p = multiprocessing.Pool(multiprocessing.cpu_count())

        results = p.starmap(random_rollout, [(actions, envs[i]) for i in range(len(envs))])
        p.close()

        return np.mean(results)


class PFAgent(object):
    def __init__(self, n_particles, game, game_params, gamma):
        self.gamma = gamma
        self.envs = []
        self.states = []
        for _ in n_particles:
            env = make_game(game, game_params)
            env.seed(np.random.randint(1e7))  # draw some Env seed
            s = env.reset()
            self.envs.append(env)
            self.states.append(s)

    def step(self):
        new_particles = []
        bp = False
        rewards = [0]
        for i in range(len(self.envs)):
            s, r, done, _ = self.envs[i].step()
            new_particles.append(s)
            if done:
                rewards.append(r)
                bp = True

        if bp:
            self.backpropagate(np.mean(rewards))

    def backpropagate(self, avg_reward):
        pass

