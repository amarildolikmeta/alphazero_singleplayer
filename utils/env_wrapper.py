import copy
import random
import signal

import numpy as np

from helpers import argmax


class TimedOutExc(Exception):
    pass


def signal_handler(signum, frame):
    raise TimedOutExc("Timed out!")


class Wrapper(object):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                 temp, game_maker=None, env=None, mcts_only=True, scheduler_params=None):

        assert game_maker is not None or env is not None, "No environment or maker provided to the wrapper"

        self.root_index = root_index
        self.env = env
        self.mcts_only = mcts_only
        self.episode_probabilities = []
        self.curr_probs = None
        self.starting_states = []
        self.model_file = model_save_file
        self.model_wrapper_params = model_wrapper_params
        self.mcts_maker = mcts_maker
        self.mcts_params = mcts_params
        self.game_maker = game_maker
        self.model = None
        self.mcts = None
        # self.action_dim = Env.action_space.n
        self.is_atari = is_atari
        self.n_mcts = n_mcts
        self.budget = budget
        self.mcts_env = mcts_env
        self.mcts_only = mcts_only
        self.c_dpw = c_dpw
        self.temp = temp
        self.scheduler_params = scheduler_params
        self.scheduler_budget = np.inf

        if not self.is_atari:
            self.mcts_env = None

    @staticmethod
    def schedule(x, k=1, width=1, mid=0):
        # if x == 0:
        #     return 0
        # elif x == width:
        #     return 1

        # width = float(width)
        # norm_x = x / width
        # parenth = norm_x / (1 - norm_x)
        # denom = 1 + parenth ** -k
        # return 1/denom
        max_depth = float(width)
        return (1 - 5 / max_depth) ** (x)

    def pi_wrapper(self, s, current_depth, max_depth):
        # Compute the reduced budget as function of the search root depth
        if self.scheduler_params:
            l = self.schedule(current_depth,
                              k=self.scheduler_params["slope"],
                              mid=self.scheduler_params["mid"],
                              width=current_depth + max_depth)
            # self.scheduler_budget = max(int(self.budget * (1 - l)), self.scheduler_params["min_budget"])
            self.scheduler_budget = max(int(self.budget * l), self.scheduler_params["min_budget"])
            # print("\nDepth: {}\nBudget: {}".format(current_depth, self.scheduler_budget))

        if self.mcts_only:
            self.search(self.n_mcts, self.c_dpw, self.mcts_env, max_depth)
            state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled
            self.curr_probs.append(pi)
            a_w = argmax(pi)
            # max_p = np.max(pi)
            # a_w = np.random.choice(np.argwhere(pi == max_p)[0])
        else:
            pi_w = self.get_model().predict_pi(s).flatten()
            self.curr_probs.append(pi_w)
            max_p = np.max(pi_w)
            a_w = np.random.choice(np.argwhere(pi_w == max_p)[0])
        return a_w

    def get_env(self):
        if self.env is None:
            self.make_env()
        return self.env

    def get_model(self):
        if not self.model:
            pass
            # self.model = ModelWrapper(**self.model_wrapper_params)
            # self.model.load(self.model_file)
        return self.model

    def get_mcts(self):
        if not self.mcts:
            self.make_mcts()
        return self.mcts

    def make_mcts(self):
        self.mcts = self.mcts_maker(root_index=self.root_index, root=None, model=self.get_model(),
                                    na=self.env.action_space.n, **self.mcts_params)

    def make_env(self):
        if self.game_maker is None:
            pass
        else:
            builder = self.game_maker["game_maker"]
            game = self.game_maker["game"]
            game_params = self.game_maker["game_params"]
            self.env = builder(game, game_params)
            seed = random.randint(0, 1e7)  # draw some Env seed
            # print("Random seed:", seed)
            self.env.seed(seed)
            self.env.reset()

    def visualize(self):
        self.get_mcts().visualize()

    def render(self):
        try:
            env = self.render_env
        except:
            env = copy.deepcopy(self.get_env())
            self.render_env = env
        env.set_signature(self.get_env().get_signature())
        env._render()

    def reset(self):
        s = self.get_env().reset()
        self.make_mcts()
        self.starting_states.append(s)
        if self.curr_probs is not None:
            self.episode_probabilities.append(self.curr_probs)
        self.curr_probs = []
        return s

    def forward(self, a, s, r):
        if self.mcts_only:
            self.get_mcts().forward(a, s, r)

    def step(self, a):
        return self.get_env().step(a)

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts,
                               c=c_dpw,
                               env=self.get_env(),
                               mcts_env=mcts_env,
                               max_depth=max_depth,
                               budget=min(self.budget, self.scheduler_budget))
        # self.get_mcts().visualize()

    def return_results(self, temp):
        return self.get_mcts().return_results(temp=temp)
