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
            self._agents_count = len(active_drivers)
            f.close()
        assert len(mcts_maker) == len(mcts_params) == self._agents_count, "Mismatch in number of agents and config data"

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
            self.search(self.n_mcts, self.c_dpw[self._current_agent], self.mcts_env, max_depth)
            state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled
            self.curr_probs.append(pi)
            a_w = argmax(pi)
            # max_p = np.max(pi)
            # a_w = np.random.choice(np.argwhere(pi == max_p)[0])
        else:
            raise NotImplementedError("No policy network has been implemented for RaceStrategy")
        return a_w

    def get_mcts(self):
        if not self.mcts:
            self.mcts = [self.make_mcts() for _ in range(self._agents_count)]
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
        print("Agent {}, action {}".format(agent, a))
        s, r, done, _ = self.get_env().partial_step(a, agent)
        self._current_agent = self.get_env().get_next_agent()
        print("Next agent:", self._current_agent)
        print()
        return s, r, done, {}

    def reset(self):
        s = self.get_env().reset()
        self.make_mcts()
        self.starting_states.append(s)
        if self.curr_probs is not None:
            self.episode_probabilities.append(self.curr_probs)
        self.curr_probs = []
        self._current_agent = self.get_env().get_next_agent()

        return s

    def make_mcts(self):
        self.mcts = [self.mcts_maker[i](root_index=self.root_index,
                                        root=None,
                                        model=self.get_model(),
                                        na=self.env.action_space.n,
                                        **self.mcts_params[i],
                                        owner=i) for i in range(self._agents_count)]
