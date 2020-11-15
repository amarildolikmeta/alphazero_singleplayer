import copy
import random
import signal
from datetime import datetime

import numpy as np

from envs.planning_env import PlanningEnv
from helpers import argmax
from utils.logging import Logger


class TimedOutExc(Exception):
    pass


def signal_handler(signum, frame):
    raise TimedOutExc("Timed out!")


class Wrapper(object):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c,
                 temp, game_maker=None, env=None, mcts_only=True, scheduler_params=None,
                 enable_logging=True, verbose=True, visualize=False):

        assert game_maker is not None or env is not None, "No environment or maker provided to the wrapper"

        self._schedules=[]
        self._depths=[]

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
        self.c = c
        self.temp = temp
        self.scheduler_params = scheduler_params
        self.scheduler_budget = np.inf

        if not self.is_atari:
            self.mcts_env = None

        self.enable_logging=enable_logging
        self.verbose = verbose
        self.visualize_search_tree = visualize

    @staticmethod
    def schedule(x, k=1, width=1, min_depth=1) -> float:
        """The method computes a reduction factor for the budget to use during search.
        The intuition is that it requires less budget to build the full tree when next to the final states,
        so the search would take lots of time.

        """

        if x <= min_depth:
            return 1
        elif x == width:
            return 0

        x -= min_depth

        width = float(width)
        norm_x = x / width
        parenth = norm_x / (1 - norm_x)
        denom = 1 + parenth ** k
        return 1 / denom

    # @staticmethod
    # def schedule(x, k=1, width=1, min_depth=1):
    #     # if x == 0:
    #     #     return 0
    #     # elif x == width:
    #     #     return 1
    #
    #     # width = float(width)
    #     # norm_x = x / width
    #     # parenth = norm_x / (1 - norm_x)
    #     # denom = 1 + parenth ** -k
    #     # return 1/denom
    #     max_depth = float(width)
    #     return (1 - 5 / max_depth) ** (x)

    def pi_wrapper(self, s, current_depth, max_depth):

        width = min(self.get_env().get_max_ep_length(), max_depth+current_depth)

        # Compute the reduced budget as function of the search root depth
        if self.scheduler_params:
            l = self.schedule(current_depth,
                              k=self.scheduler_params["slope"],
                              min_depth=self.scheduler_params["min_depth"],
                              width=width)
            # self.scheduler_budget = max(int(self.budget * (1 - l)), self.scheduler_params["min_budget"])
            self.scheduler_budget = max(int(self.budget * l), self.scheduler_params["min_budget"])

            # print("\nDepth: {}\nBudget: {}".format(current_depth, self.scheduler_budget))

        if self.mcts_only:
            # TODO remove, only for raceStrategy debugging
            owner = self.get_env().get_next_agent()
            actions = self.get_env().get_available_actions(owner)

            if len(actions) > 1:
                self.search(self.n_mcts, self.c, self.mcts_env, max_depth)
                state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled

                if not len(actions) == self.get_env().action_space.n:
                    # in this case pi is just the policy over the compacted action list,
                    # need to remap to full action space
                    fixed_pi = np.zeros(self.get_mcts().na)
                    actions = self.get_env().get_available_actions(owner)
                    for i in range(len(pi)):
                        action_index = actions[i]
                        fixed_pi[action_index] = pi[i]  # Set the probability mass only for those actions that are available
                    pi = fixed_pi

            else:  # Only one action can be performed, skip search
                pi = np.zeros(self.get_mcts().na)
                pi[0] = 1.
            self.curr_probs.append(pi)
            a_w = argmax(pi)
            if self.verbose:
                print("Search result:")
                print("Policy: {},\nWinning action: {}".format(pi, a_w))
                # max_p = np.max(pi)
                # a_w = np.random.choice(np.argwhere(pi == max_p)[0])

        else:
            pi_w = self.get_model().predict_pi(s).flatten()
            self.curr_probs.append(pi_w)
            max_p = np.max(pi_w)
            a_w = np.random.choice(np.argwhere(pi_w == max_p)[0])
        return a_w

    def get_env(self) -> PlanningEnv:
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
        if self.verbose:
            print("Action:", a)
            print()
        s, r, done, _ = self.get_env().step(a)
        if done and self.enable_logging:
            self.get_env().save_results(Logger().timestamp)
        return s, r, done, _

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts,
                               c=c_dpw,
                               Env=self.get_env(),
                               mcts_env=mcts_env,
                               max_depth=max_depth,
                               budget=min(self.budget, self.scheduler_budget))
        if self.visualize_search_tree:
            self.get_mcts().visualize()

    def return_results(self, temp):
        return self.get_mcts().return_results(temp=temp)
