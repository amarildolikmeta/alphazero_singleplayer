import copy
import random
import signal
from datetime import datetime

import numpy as np

from helpers import argmax


class TimedOutExc(Exception):
    pass


def signal_handler(signum, frame):
    raise TimedOutExc("Timed out!")


class Wrapper(object):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                 temp, game_maker=None, env=None, mcts_only=True, scheduler_params=None,
                 log_path="./logs/", log_timestamp=None):

        assert game_maker is not None or env is not None, "No environment or maker provided to the wrapper"

        self.agents_count = 1
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

        # Set the timestamp
        if not log_timestamp:
            today = datetime.now()
            self.timestamp = today.strftime('%Y-%m-%d_%H-%M')
        else:
            assert type(log_timestamp) == str, "Timestamp must be provided as string"
            self.timestamp = log_timestamp

        self.timestamp = self.timestamp + "_" + str(budget) + "b"

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
            # TODO remove, only for raceStrategy debugging
            if hasattr(self.get_env(), "get_available_actions") and hasattr(self.get_env(), "get_next_agent"):
                owner = self.get_env().get_next_agent()
                if len(self.get_env().get_available_actions(owner)) > 1:
                    self.search(self.n_mcts, self.c_dpw, self.mcts_env, max_depth)
                    state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled
                    # This is just the policy over the compacted action list, need to remap to full action space
                    fixed_pi = np.zeros(self.get_mcts().na)
                    actions = self.get_env().get_available_actions(self.get_mcts().owner)
                    for i in range(len(pi)):
                        action_index = actions[i]
                        fixed_pi[action_index] = pi[i]  # Set the probability mass only for those actions that are available
                    pi = fixed_pi
                else:  # Pit-stop cannot be performed, skip the search and default to stay on track action
                    pi = np.zeros(self.get_mcts().na)
                    pi[0] = 1.
                self.curr_probs.append(pi)
                a_w = argmax(pi)
                # max_p = np.max(pi)
                # a_w = np.random.choice(np.argwhere(pi == max_p)[0])
            else:
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
        s, r, done, _ = self.get_env().step(a)
        if done and self.game_maker["game"] == 'RaceStrategy-v2':
            self.get_env().save_results(self.timestamp)
        return self.get_env().step(a)

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts,
                               c=c_dpw,
                               Env=self.get_env(),
                               mcts_env=mcts_env,
                               max_depth=max_depth,
                               budget=min(self.budget, self.scheduler_budget))
        # self.get_mcts().visualize()

    def return_results(self, temp):
        return self.get_mcts().return_results(temp=temp)
