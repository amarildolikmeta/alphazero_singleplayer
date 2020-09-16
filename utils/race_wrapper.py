import csv
import numpy as np

from utils.env_wrapper import Wrapper

from helpers import argmax


class RaceWrapper(Wrapper):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                 temp, game_maker=None, env=None, mcts_only=True, scheduler_params=None):
        super(RaceWrapper, self).__init__(root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                 temp, game_maker, env, mcts_only, scheduler_params)

        with open('./envs/race_strategy_model/active_drivers.csv', newline='') as f:
            reader = csv.reader(f)
            line = reader.__next__()
            active_drivers = np.asarray(line[1:], dtype=int)
            self._drivers_count = len(active_drivers)
            f.close()

        self.agents_queue = self.mcts_env.get_agents_standings()
        self._current_agent = self.agents_queue.get()

    def get_mcts(self):
        if not self.mcts:
            self.mcts = [self.make_mcts() for _ in range(self._drivers_count)]
        return self.mcts[self._current_agent]

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts,
                               c=c_dpw,
                               env=self.get_env(),
                               mcts_env=mcts_env,
                               max_depth=max_depth,
                               budget=min(self.budget, self.scheduler_budget))
        # self.get_mcts().visualize()

    def step(self, a):
        agent = self._current_agent
        self._current_agent = self.agents_queue.get()
        if self.get_env().has_transitioned():
            self.agents_queue = self.get_env().get_agents_standings()
        return self.get_env().partial_step(a, agent)

    def reset(self):
        s = self.get_env().reset()
        self.make_mcts()
        self.starting_states.append(s)
        if self.curr_probs is not None:
            self.episode_probabilities.append(self.curr_probs)
        self.curr_probs = []

        self.agents_queue = self.get_env().get_agents_standings()
        self._current_agent = self.agents_queue.get()

        return s

    def make_mcts(self):
        self.mcts = [self.mcts_maker[i](root_index=self.root_index, root=None, model=self.get_model(),
                                    na=self.env.action_space.n, **self.mcts_params) for i in range(self._drivers_count)]

