import csv
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from utils.env_wrapper import Wrapper
from envs.race_strategy_full import compute_ranking
from helpers import argmax

LOG_COLUMNS = ["driver", "agent", "turn", "race", "total_time", "starting_position", "final_position", "pit_count"]

class RaceWrapper(Wrapper):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                 temp, game_maker=None, env=None, mcts_only=True, scheduler_params=None,
                 enable_logging=False, log_path="./logs/", log_timestamp=None, verbose=True):

        super(RaceWrapper, self).__init__(root_index, mcts_maker, model_save_file, model_wrapper_params,
                                          mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                                          temp, game_maker, env, mcts_only, scheduler_params, log_path, log_timestamp)

        self.verbose = verbose
        # Create the log folder
        if enable_logging:
            os.mkdir(self.log_path)
            self.log_path += "/race_log_{}b.csv".format(budget)
        self.experiment_counter = 0

        # Load the active drivers
        with open('./envs/race_strategy_model/active_drivers.csv', newline='') as f:
            reader = csv.reader(f)
            line = reader.__next__()
            self.active_drivers = np.asarray(line[1:], dtype=int)
            self.agents_count = len(self.active_drivers)
            f.close()
        assert len(mcts_maker) == len(mcts_params) == self.agents_count, "Mismatch in number of agents and config data"

        self.start = self.get_env().start_lap
        self._race_length = self.get_env().get_race_length()
        self.t = 0
        self.logs = []
        self.log_dataframe = pd.DataFrame()
        self.enable_logging = enable_logging
        self.pit_counts = [0] * self.agents_count
        self.index = 0

        self._current_agent = self.get_env().get_next_agent()

    @staticmethod
    def schedule(x, k=1, width=1, min_depth=1) -> float:
        """The method computes a reduction factor for the budget tu use during search.
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
        return 1/denom

    def pi_wrapper(self, s, current_depth, max_depth):
        # Compute the reduced budget as function of the search root depth
        if self.scheduler_params:
            l = self.schedule(current_depth + self.start,
                              k=self.scheduler_params["slope"],
                              min_depth=self.scheduler_params["min_depth"],
                              width=self._race_length)
            # self.scheduler_budget = max(int(self.budget * (1 - l)), self.scheduler_params["min_budget"])
            self.scheduler_budget = max(int(self.budget * l), self.scheduler_params["min_budget"])
            # print("\nDepth: {}\nBudget: {}".format(current_depth, self.scheduler_budget))

        if self.mcts_only:
            if len(self.get_env().get_available_actions(self.get_mcts().owner)) > 1:
                self.search(self.n_mcts, self.c_dpw[self._current_agent], self.mcts_env, max_depth)
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
            raise NotImplementedError("No policy network has been implemented for RaceStrategy")
        return a_w

    def get_mcts(self):
        if not self.mcts:
            self.mcts = [self.make_mcts() for _ in range(self.agents_count)]
        return self.mcts[self._current_agent]

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts,
                               c=c_dpw,
                               env=self.get_env(),
                               mcts_env=mcts_env,
                               max_depth=max_depth,
                               budget=min(self.budget, self.scheduler_budget))
        # self.get_mcts().visualize()

    def step(self, a) -> tuple:
        agent = self._current_agent
        if self.enable_logging:
            if a > 0:
                self.pit_counts[agent] += 1
        if self.verbose:
            print("Lap {}: Agent {}, action {}".format(self.start + self.t, agent, a))
        s, r, done, _ = self.get_env().partial_step(a, agent)

        self._current_agent = self.get_env().get_next_agent()
        if self.get_env().has_transitioned():
            self.t += 1
        # print("Next agent:", self._current_agent)
        if self.verbose:
            print()
        if done:
            self.get_env().save_results(self.timestamp)
            if self.enable_logging:
                self.finalize_logs()
        return s, r, done, {}

    def reset(self):
        # Write the logs if any are present, then store initial conditions
        self.experiment_counter += 1
        if self.enable_logging:
            self.write_log()
        s = self.get_env().reset()
        self.start = self.get_env().start_lap
        self._race_length = self.get_env().get_race_length()

        self.make_mcts()
        self.starting_states.append(s)
        if self.curr_probs is not None:
            self.episode_probabilities.append(self.curr_probs)
        self.curr_probs = []
        self._current_agent = self.get_env().get_next_agent()
        self.start = self.get_env().start_lap
        self.t = 0
        self.pit_counts = [0] * self.agents_count

        return s

    def make_mcts(self):
        self.mcts = [self.mcts_maker[i](root_index=self.root_index,
                                        root=None,
                                        na=self.env.action_space.n,
                                        **self.mcts_params[i],
                                        owner=i) for i in range(self.agents_count)]

    def write_log(self):
        """Log pandas dataframe"""
        if len(self.logs) > 0:
            for log in self.logs:
                log["experiment"] = self.experiment_counter # A race might repeat, so we need to distinguish the data
                df = pd.DataFrame(log, index=[self.index])
                self.index += 1
                self.log_dataframe = self.log_dataframe.append(df)
            self.log_dataframe.to_csv(self.log_path)

    def finalize_logs(self):
        final_log = self.get_env().get_log_info()
        for i in range(len(final_log)):
            if final_log[i]['driver'] not in self.active_drivers:
                final_log[i]['agent'] = "passive"
                final_log[i]["turn"] = -1
            else:
                turn = np.argwhere(self.active_drivers == final_log[i]["driver"])[0][0]
                final_log[i]["pit_count"] = self.pit_counts[turn]
                final_log[i]["agent"] = self.mcts[turn].NAME
                final_log[i]["turn"] = turn
        self.logs = final_log

