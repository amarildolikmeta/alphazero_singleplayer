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
                 enable_logging=False, log_path="./logs/Racestrategy-full/", log_timestamp=None, verbose=False):

        super(RaceWrapper, self).__init__(root_index, mcts_maker, model_save_file, model_wrapper_params,
                                          mcts_params, is_atari, n_mcts, budget, mcts_env, c_dpw,
                                          temp, game_maker, env, mcts_only, scheduler_params)


        self.verbose = verbose
        # Create the log folder
        if not log_timestamp:
            today = datetime.now()
            self.timestamp = today.strftime('%Y-%m-%d_%H-%M')
        else:
            assert type(log_timestamp) == str, "Timestamp must be provided as string"
            self.timestamp = log_timestamp

        self.timestamp = self.timestamp + "_" + str(budget) + "b"
        self.log_path = log_path + self.timestamp

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
        self.t = 0
        self.logs = []
        self.log_dataframe = pd.DataFrame()
        self.enable_logging = enable_logging
        self.pit_counts = [0] * self.agents_count
        self.index = 0

        self._current_agent = self.get_env().get_next_agent()

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
            if len(self.get_env().get_available_actions(self.get_mcts().owner)) > 1:
                self.search(self.n_mcts, self.c_dpw[self._current_agent], self.mcts_env, max_depth)
                state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled
                #print(pi)
            else:
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

    def step(self, a):
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
                                        model=self.get_model(),
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

