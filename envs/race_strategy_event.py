import os
import sys
import csv
import re
from collections import deque, defaultdict
from os import listdir
from os.path import isfile, join
import random
import gym
from copy import deepcopy
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register
import pandas as pd

# Since the module race_simulation was forked it is not configured to work with the repository's folder structure
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_PATH, 'race_simulation'))

from race_simulation.racesim.src.race import Race
from race_simulation.racesim.src.import_pars import import_pars
from race_simulation.racesim.src.check_pars import check_pars

MCS_PARS_FILE = 'pars_mcs.ini'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)


def generate_race_event(**game_params):
    if game_params is None:
        game_params = {}
    return RaceEnv(**game_params)


def compute_ranking(cumulative_time):
    """Computes the ranking on track, based on cumulative times"""
    temp = np.argsort(cumulative_time)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(cumulative_time)) + 1
    return ranks


def select_races(year: int, ini_path="./race_simulation/racesim/input/parameters") -> list:
    regex = re.compile('.*_.*_{}\.ini'.format(year))
    onlyfiles = [f for f in listdir(ini_path) if isfile(join(ini_path, f))]
    final = []
    for file in onlyfiles:
        if regex.match(file):
            final.append(file)
    return final


class RaceEnv(gym.Env):

    def __init__(self, gamma=0.95, horizon=20, scale_reward=False, positive_reward=True, start_lap=8,
                 verbose=False, config_path='./envs/race_strategy_model/active_drivers.csv', n_cores=-1):

        self.verbose = verbose
        self._actions_queue = deque()
        self._agents_queue = deque()
        self._agents_last_pit = defaultdict(int)
        self.horizon = horizon
        self.gamma = gamma
        self.obs_dim = 7
        self.action_space = spaces.Discrete(n=4)
        self.observation_space = spaces.Box(low=0., high=self.horizon,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.scale_reward = scale_reward
        self.positive_reward = positive_reward
        self.viewer = None
        self.step_id = 0

        # get repo path
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # create output folders (if not existing)
        self.output_path = os.path.join(repo_path, "logs", "RaceStrategy-v2")

        self.results_path = os.path.join(self.output_path, "results")
        os.makedirs(self.results_path, exist_ok=True)

        self.invalid_dumps_path = os.path.join(self.output_path, "invalid_dumps")
        os.makedirs(self.invalid_dumps_path, exist_ok=True)

        self.testobjects_path = os.path.join(self.output_path, "testobjects")
        os.makedirs(self.testobjects_path, exist_ok=True)

        with open(config_path, newline='') as f:
            reader = csv.reader(f)
            line = reader.__next__()
            self._active_drivers = np.asarray(line[1:], dtype=int)
            self._year = int(line[0])
            f.close()

        self._active_drivers_mapping = {}
        self._index_to_active = {}
        for i in range(len(self._active_drivers)):
            self._active_drivers_mapping[self._active_drivers[i]] = i
            self._index_to_active[i] = self._active_drivers[i]

        self._active_drivers = set(self._active_drivers)
        self.agents_number = len(self._active_drivers)

        self.start_lap = start_lap
        self._lap = 0
        self.race_length = 0

        self._t = -start_lap

        self._races_config_files = select_races(self._year)
        random.shuffle(self._races_config_files)

        self._compound_initials = []
        self._race_sim = None
        # TODO discard wet races

        # df = pd.DataFrame(columns=self._model.dummy_columns)
        # df['step'] = None
        # df.to_csv('./logs/pred_log.csv')

        self._drivers_number = 0

        # Take the base time, the predictions will be deltas from this time
        self.max_lap_time = 600
        self._drivers = []
        self._race_length = 0

        self._lap_time = None
        self._cumulative_time = None

        self._drivers_mapping = {}
        self._index_to_driver = {}

        self._terminal = False
        self._compound_initials = []

        self.seed()

        if self.scale_reward and verbose:
            print("Reward is being normalized")

    def get_state(self) -> dict:
        state = self._race_sim.get_simulation_state()
        return state

    def __set_state(self, state):
        pass

    def get_signature(self):
        sig = {'state': deepcopy(self.get_state()),
               'action_queue': deepcopy(self._actions_queue),
               'agents_queue': deepcopy(self._agents_queue),
               'last_pits': deepcopy(self._agents_last_pit),
               'simulator': deepcopy(self._race_sim)
               }
        return sig

    def set_signature(self, sig):
        self.__set_state(sig["state"])
        self._race_sim = sig['simulator']
        self._actions_queue = sig['action_queue']
        self._agents_queue = sig['agents_queue']
        self._agents_last_pit = sig['last_pits']

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_available_actions(self, agent: int) -> list:
        """Allow another pit stop, signified by actions 1-3, only if the last pit
        for the same agents was at least 5 laps earlier"""

        actions = [0]
        if self._agents_last_pit[agent] > 5:
            actions.extend([i for i in range(1, len(self._compound_initials) + 1)])
        return actions

    def map_action_to_compound(self, action_index: int) -> str:
        assert len(self._compound_initials) > 0, "[ERROR] Env has not been reset yet"
        if action_index == 0:
            return ""
        else:
            return self._compound_initials[action_index - 1]

    def partial_step(self, action, owner):
        """Accept an action for an gent, but perform the model transition only when an action for each agent has
        been specified"""

        agent = self.get_next_agent()
        self._agents_queue.popleft()
        assert owner == agent, "Actions are de-synchronized with agents queue {} {}".format(owner, agent)

        self._actions_queue.append((action, owner))
        if action == 0:  # No pit, increment the time from last pit
            self._agents_last_pit[owner] += 1
        else:  # Pit, reset time
            self._agents_last_pit[owner] = 0

        if len(self._actions_queue) == self.agents_number:
            actions = np.zeros(self.agents_number, dtype=int)
            while not len(self._actions_queue) == 0:
                action, owner_index = self._actions_queue.popleft()
                actions[owner_index] = action
            return self.step(actions)

        else:
            return self.get_state(), np.zeros(self.agents_number), self._terminal, {}

    def has_transitioned(self):
        return len(self._actions_queue) == 0

    def step(self, actions: np.ndarray):
        assert self._race_sim is not None, "[ERROR] You tried to perform a step in the environment before resetting it"
        self.step_id += 1

        self._lap = self._race_sim.get_cur_lap()

        if self._t >= self.horizon or self._lap >= self.race_length:
            return self.get_state(), [0] * self.agents_number, True, {}

        pit_info = []
        for action, idx in zip(actions, range(len(actions))):
            if action > 0:
                compound = self.map_action_to_compound(action)
                pit_info.append((self._index_to_active[idx], [compound, 0, 0.]))
        predicted_times, driver_info = self._race_sim.step(pit_info)

        lap_times = np.ones(self._drivers_number) * self.max_lap_time
        for lap_time, driver in zip(predicted_times, driver_info):
            index = self._drivers_mapping[driver.carno]
            lap_times[index] = lap_time
            #self._cumulative_time[index] += lap_time

        reward = np.ones(self.agents_number) * self.max_lap_time
        for driver in self._active_drivers:
            active_index = self._active_drivers_mapping[driver]
            index = self._drivers_mapping[driver]
            reward[active_index] = -np.clip(lap_times[index], 0, self.max_lap_time)

        if self.scale_reward:
            reward /= self.max_lap_time
            if self.positive_reward:
                reward = 1 + reward

        self._t += 1
        self._lap = self._race_sim.get_cur_lap()
        self._terminal = True if self._t >= self.horizon or self._lap >= self.race_length else False

        return self.get_state(), reward, self._terminal, {}

    def save_results(self, timestamp):
        save_path = os.path.join(self.results_path, timestamp)
        os.makedirs(save_path, exist_ok=True)
        self._race_sim.export_results_as_csv(results_path=save_path)

    def reset(self):

        # set simulation options
        # use_prob_infl:        activates probabilistic influences within the race simulation -> lap times, pit stop
        #                       durations, race start performance
        # create_rand_events:   activates the random creation of FCY (full course yellow) phases and retirements in the race
        #                       simulation -> they will only be created if the according entries in the parameter file
        #                       contain empty lists, otherwise the file entries are used
        # use_vse:              determines if the VSE (virtual strategy engineer) is used to take tire change decisions
        #                       -> the VSE type is defined in the parameter file (VSE_PARS)
        # no_sim_runs:          number of (valid) races to simulate
        # no_workers:           defines number of workers for multiprocess calculations, 1 for single process, >1 for
        #                       multi-process (you can use print(multiprocessing.cpu_count()) to determine the max. number)
        # use_print:            set if prints to console should be used or not (does not suppress hints/warnings)
        # use_print_result:     set if result should be printed to console or not
        # use_plot:             set if plotting should be used or not
        self._t = -self.start_lap
        self._terminal = False
        race_pars_file = self._races_config_files.pop(0)
        print(race_pars_file)
        self._races_config_files.append(race_pars_file)

        sim_opts = {"use_prob_infl": True,
                    "create_rand_events": False,
                    "use_vse": False,
                    "no_sim_runs": 1,
                    "no_workers": 1,
                    "use_print": False,
                    "use_print_result": False,
                    "use_plot": False}


        # load parameters
        pars_in, vse_paths = import_pars(use_print=sim_opts["use_print"],
                                                                 use_vse=sim_opts["use_vse"],
                                                                 race_pars_file=race_pars_file,
                                                                 mcs_pars_file=MCS_PARS_FILE)

        # check parameters
        check_pars(sim_opts=sim_opts, pars_in=pars_in)

        self._race_sim = Race(race_pars=pars_in["race_pars"],
                              driver_pars=pars_in["driver_pars"],
                              car_pars=pars_in["car_pars"],
                              tireset_pars=pars_in["tireset_pars"],
                              track_pars=pars_in["track_pars"],
                              vse_pars=pars_in["vse_pars"],
                              vse_paths=vse_paths,
                              use_prob_infl=sim_opts['use_prob_infl'],
                              create_rand_events=sim_opts['create_rand_events'],
                              monte_carlo_pars=pars_in["monte_carlo_pars"],
                              event_pars=pars_in["event_pars"],
                              disable_retirements=True)

        self._compound_initials = pars_in["vse_pars"]["param_dry_compounds"]
        print(self._compound_initials)
        state = self._race_sim.get_simulation_state()

        # Initialize the driver number / array indices mapping
        self._drivers = [driver.carno for driver in state["drivers"]]
        self._drivers_number = len(self._drivers)
        for i in range(self._drivers_number):
            self._drivers_mapping[self._drivers[i]] = i
            self._index_to_driver[i] = self._drivers[i]

        # Use the default strategies before the actual start
        while self._race_sim.get_cur_lap() < self.start_lap:
            self._race_sim.step([])

        self._race_sim.set_controlled_drivers(list(self._active_drivers))
        self.race_length = self._race_sim.get_race_length()

        return self.get_state()

    def get_agents_standings(self):
        temp = np.argsort(self._cumulative_time)
        ranks = deque()
        for index in temp:
            driver = self._index_to_driver[index]
            if driver in self._active_drivers:
                ranks.append(self._active_drivers_mapping[driver])
        return ranks

    def get_next_agent(self):
        if len(self._agents_queue) == 0:
            for i in range(self.agents_number):
                self._agents_queue.append(i)
            # self._agents_queue = self.get_agents_standings()
        return self._agents_queue[0]

    def is_terminal(self):
        return self._terminal

    def get_log_info(self):
        logs = []
        ranks = compute_ranking(self._cumulative_time)
        for driver in self._drivers:
            index = self._drivers_mapping[driver]
            log = {
                "position": ranks[index],
                "cumulative_time": self._cumulative_time[index],
                "pit_count": self._pit_counts[index],
                "driver": driver,
                "race": self.get_current_race(),
                "start_lap": self.start_lap,
                "current_lap": self._lap,
                "starting_position": self._starting_positions[index]
            }
            logs.append(log)

        return logs


register(
    id='RaceStrategy-v2',
    entry_point='envs.race_strategy_full:RaceModel'
)

if __name__ == '__main__':
    mdp = RaceEnv()
    _ = mdp.reset()
    ret = 0
    lap = 1
    while True:
        a = np.random.choice([0, 1, 2, 3], 9, replace=True)
        s, r, done, _ = mdp.step(a)
        print("Reward:" + str(r) + " Lap Time: " + str(r * mdp.max_lap_time))
        mdp.set_signature(mdp.get_signature())
        ret += r
        if done:
            print("Return:", ret)
            # print("Race Time:", mdp.time)
            break
