import csv
from collections import deque, defaultdict

import gym
from copy import deepcopy
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register
from envs.race_strategy_model.prediction_model import RaceStrategyModel
import pandas as pd

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


class RaceEnv(gym.Env):

    def __init__(self, gamma=0.95, horizon=20, scale_reward=False, positive_reward=True, start_lap=8,
                 verbose=False, n_cores=-1):

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

        with open('./envs/race_strategy_model/active_drivers.csv', newline='') as f:
            reader = csv.reader(f)
            line = reader.__next__()
            self._active_drivers = np.asarray(line[1:], dtype=int)
            self._year = int(line[0])
            f.close()

        self._active_drivers = set(self._active_drivers)
        self.agents_number = len(self._active_drivers)

        self.start_lap = start_lap
        self._lap = 0

        self._t = -start_lap

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
        race_pars_file_ = 'pars_Spielberg_2019.ini'
        mcs_pars_file_ = 'pars_mcs.ini'

        sim_opts_ = {"use_prob_infl": True,
                     "create_rand_events": False,
                     "use_vse": True,
                     "no_sim_runs": 1,
                     "no_workers": 1,
                     "use_print": False,
                     "use_print_result": False,
                     "use_plot": False}

        # get repo path
        repo_path = os.path.dirname(os.path.abspath(__file__))

        # create output folders (if not existing)
        output_path = os.path.join(repo_path, "logs", "RaceStrategy-v2")

        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        invalid_dumps_path = os.path.join(output_path, "invalid_dumps")
        os.makedirs(invalid_dumps_path, exist_ok=True)

        testobjects_path = os.path.join(output_path, "testobjects")
        os.makedirs(testobjects_path, exist_ok=True)

        # load parameters
        pars_in, vse_paths = racesim.src.import_pars.import_pars(use_print=sim_opts["use_print"],
                                                                 use_vse=sim_opts["use_vse"],
                                                                 race_pars_file=race_pars_file,
                                                                 mcs_pars_file=mcs_pars_file)

        # check parameters
        racesim.src.check_pars.check_pars(sim_opts=sim_opts, pars_in=pars_in)


        self._compound_initials = []
        self._race_sim = None

        df = pd.DataFrame(columns=self._model.dummy_columns)
        df['step'] = None
        df.to_csv('./logs/pred_log.csv')

        self._drivers_number = 0

        # Take the base time, the predictions will be deltas from this time
        self.max_lap_time = 0
        self._drivers = None
        self._race_length = 0
        self._default_strategies = None
        self._laps = None

        self._pit_counts = None
        self._tyre_age = None
        self._current_tyres = None

        self._old_lap_time = None
        self._lap_time = None
        self._cumulative_time = None
        self._next_lap_time = None
        self._lap_deltas = None
        self._last_available_row = None

        self._soft_tyre, self._medium_tyre, self._hard_tyre = None, None, None

        self._drivers_mapping = {}
        self._index_to_driver = {}
        self._active_drivers_mapping = {}

        self._terminal = False
        self._reward = np.zeros(self.agents_number)
        self._starting_positions = None

        self.seed()
        self.reset()

        if self.scale_reward and verbose:
            print("Reward is being normalized")

    def get_state(self):
        # start = time.time()
        state = [deepcopy(self._lap),
                 deepcopy(self._current_tyres),
                 deepcopy(self._cumulative_time),
                 deepcopy(self._lap_time),
                 deepcopy(self._pit_states),
                 deepcopy(self._pit_counts),
                 deepcopy(self._tyre_age)]
        # stop = time.time()
        # print("get state time:", stop - start)
        return state

    def __set_state(self, state):
        self._lap = state[0]
        self._t = self._lap - self.start_lap
        self._current_tyres = state[1]
        self._cumulative_time = state[2]
        self._lap_time = state[3]
        self._pit_states = state[4]
        self._pit_counts = state[5]
        self._tyre_age = state[6]

    def get_signature(self):
        sig = {'state': deepcopy(self.get_state()),
               'next_lap_time': deepcopy(self._next_lap_time),
               'last_row': deepcopy(self._last_available_row),
               'action_queue': deepcopy(self._actions_queue),
               'agents_queue': deepcopy(self._agents_queue),
               'last_pits': deepcopy(self._agents_last_pit)
               }
        return sig

    def set_signature(self, sig):
        self.__set_state(sig["state"])
        self._next_lap_time = sig['next_lap_time']
        self._last_available_row = sig['last_row']
        self._actions_queue = sig['action_queue']
        self._agents_queue = sig['agents_queue']
        self._agents_last_pit = sig['last_pits']

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __pit_driver(self, driver, tyre, safety):
        index = self._drivers_mapping[driver]
        self._pit_states[index] = -1
        self._current_tyres[index] = tyre
        self._pit_counts[index] += 1
        condition = 'safety' if safety else 'regular'
        self._pit_costs[index] = np.random.normal(self._model.test_race_pit_model[condition][0],
                                                  self._model.test_race_pit_model[condition][1])

    def __update_pit_flags(self, index):
        if self._pit_states[index] == -1:
            self._pit_states[index] = 1
            self._tyre_age[index] = -1

        elif self._pit_states[index] == 1:
            self._pit_states[index] = 0
            self._pit_costs[index] = 0

        else:
            self._pit_costs[index] = 0

        self._tyre_age[index] += 1

    def get_available_compounds(self):
        return self._compound_initials

    def get_available_actions(self, agent: int) -> list:
        """Allow another pit stop, signified by actions 1-3, only if the last pit
        for the same agents was at least 5 laps earlier"""
        if self._agents_last_pit[agent] > 5:
            return [i for i in range(len(self._compound_initials))]
        else:
            return [0]

    def map_action_to_compound(self, action_index) -> str:
        if action_index == 0:
            return ""
        else:
            return self._compound_initials[action_index]

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
            actions = np.zeros(self.agents_number)
            while not len(self._actions_queue) == 0:
                action, owner_index = self._actions_queue.popleft()
                actions[owner_index] = action
            return self.step(actions)

        else:
            return self.get_state(), np.zeros(self.agents_number), self._terminal, {}

    def has_transitioned(self):
        return len(self._actions_queue) == 0

    def step(self, actions: np.ndarray):
        self.step_id += 1
        # start = time.time()
        # assert len(actions) == len(self._active_drivers), \
        #     "Fewer actions were provided than the number of active drivers"

        reward = np.zeros(len(actions))
        self._lap = self.start_lap + self._t + 1

        if self._t >= self.horizon or lap > len(self._laps):
            return self.get_state(), [0] * self.agents_number, True, {}

        pit_info = []
        for action, idx in zip(actions, range(len(actions))):
            if action > 0:
                compound = self.map_action_to_compound(action)
                pit_info.append(self.car_numbers[idx], [compound, 0, 0.])
        lap_times, driver_info = self._race_sim.step(pit_info)

        reward = np.ones(self._active_drivers) * self.max_lap_time
        for lap_time, driver in zip(lap_times, driver_info):
            index = self._drivers_mapping[driver.carno]
            reward[index] = lap_time
            self._cumulative_time[index] += lap_time

        self._lap_time = reward

        ranks = compute_ranking(self._cumulative_time)

        # safety_laps = self._model.test_race[self._model.test_race['lap'] == lap_norm]['safety'].max()

        for driver in self._active_drivers:
            active_index = self._active_drivers_mapping[driver]
            index = self._drivers_mapping[driver]
            reward[active_index] = -np.clip(self._lap_time[index], 0, self.max_lap_time)

        if self.scale_reward:
            reward /= self.max_lap_time
            if self.positive_reward:
                reward = 1 + reward

        self._t += 1
        self._terminal = True if self._t >= self.horizon or lap >= len(self._laps) else False

        # stop = time.time()
        # contribute['predict'] = stop - start
        # for k, v in contribute.items():
        #     print(k, v)

        return self.get_state(), reward, self._terminal, {}

    def reset(self):
        self._t = -self.start_lap
        self._actions_queue = deque()
        self._agents_queue = deque()
        self._agents_last_pit = defaultdict(int)
        # self._model = RaceStrategyModel(self._year)
        # self._model.load()

        if self.verbose:
            print("Looking for a race with desired drivers")
        self._model.resplit()
        self._drivers = self._model.test_race['driverId'].unique()

        # Resplit until finding a race with the selected drivers
        while not set(self._active_drivers).issubset(set(self._drivers)):
            self._model.resplit()
            self._drivers = self._model.test_race['driverId'].unique()
        if self.verbose:
            print("Found race with desired drivers")

        # Take the base time, the predictions will be deltas from this time
        self._base_time = self._model.test_race['pole'].values[0]
        self.max_lap_time = self._base_time * 5
        self._drivers_number = len(self._drivers)
        self._race_length = int(self._model.test_race['race_length'].values[0])
        self._laps = self._model.test_race.sort_values('lap')['lap'].unique()

        self._pit_states = np.zeros(self._drivers_number)
        self._pit_counts = np.zeros(self._drivers_number)
        self._pit_costs = np.zeros(self._drivers_number)
        self._tyre_age = np.zeros(self._drivers_number)
        self._current_tyres = [""] * self._drivers_number

        self._old_lap_time = np.zeros(self._drivers_number)
        self._lap_time = np.zeros(self._drivers_number)
        self._cumulative_time = np.zeros(self._drivers_number)
        self._next_lap_time = np.zeros(self._drivers_number)
        self._lap_deltas = np.zeros(self._drivers_number)
        self._last_available_row = np.empty(self._drivers_number, dtype=pd.DataFrame)

        self._soft_tyre, self._medium_tyre, self._hard_tyre = find_available_rubber(self._model.test_race)

        self._drivers_mapping = {}
        self._index_to_driver = {}
        self._active_drivers_mapping = {}

        for driver, index in zip(self._drivers, range(self._drivers_number)):
            self._drivers_mapping[driver] = index
            self._index_to_driver[index] = driver

        for driver, index in zip(self._active_drivers, range(self.agents_number)):
            self._active_drivers_mapping[driver] = index

        self._soft_tyre, self._medium_tyre, self._hard_tyre = find_available_rubber(self._model.test_race)
        tyre_average_stints = get_default_strategies(self._model.test_race)
        self._default_strategies = [None] * self._drivers_number

        for driver in self._drivers_mapping:
            index = self._drivers_mapping[driver]
            strategy = {}
            for compound in [self._soft_tyre, self._medium_tyre, self._hard_tyre]:
                compound_strategy = np.random.normal(tyre_average_stints[compound][0], tyre_average_stints[compound][1])
                if np.isinf(compound_strategy):
                    strategy[compound] = 10000  # put a large number in place of infinity
                else:
                    strategy[compound] = compound_strategy
            self._default_strategies[index] = strategy

        # Set the information for first lap

        for driver, index in zip(self._drivers, range(self._drivers_number)):
            data = self._model.test_race[(self._model.test_race['driverId'] == driver) &
                                         (self._model.test_race['unnorm_lap'] == 1)]
            if data['lap'].count() > 0:
                data = data.squeeze()
                self._cumulative_time[index] = data['cumulative']
                self._lap_time[index] = data['milliseconds']
                self._next_lap_time[index] = self._base_time + data['nextLap']
                self._pit_states[index] = data['pit']
                self._pit_counts[index] = data['stop']
                self._tyre_age[index] = data['tyre_age'] * self._race_length
                self._current_tyres[index] = get_current_tyres(data)

        # Predict the first laps for any driver
        for i in range(self.start_lap):
            self.step(np.asarray([0] * len(self._active_drivers)))

        # Set the information for prediction start lap, overwriting prediction for non-retired drivers

        for driver, index in zip(self._drivers, range(self._drivers_number)):
            data = self._model.test_race[(self._model.test_race['driverId'] == driver) &
                                         (self._model.test_race['unnorm_lap'] == self.start_lap - 1)]

            # Set information only for those drivers that have available info
            if data['lap'].count() > 0:
                data = data.squeeze()
                self._cumulative_time[index] = data['cumulative']
                self._lap_time[index] = data['milliseconds']
                self._next_lap_time[index] = self._base_time + data['nextLap']
                self._pit_states[index] = data['pit']
                self._pit_counts[index] = data['stop']
                self._tyre_age[index] = data['tyre_age'] * self._race_length
                self._current_tyres[index] = get_current_tyres(data)

        for driver in self._active_drivers:
            index = self._drivers_mapping[driver]
            agent_index = self._active_drivers_mapping[driver]
            self._agents_last_pit[agent_index] = int(self._tyre_age[index])

        self._starting_positions = compute_ranking(self._cumulative_time)

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

    def get_current_race(self):
        return self._model.race_id

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
    id='RaceStrategy-v1',
    entry_point='envs.race_strategy_full:RaceModel'
)

if __name__ == '__main__':
    mdp = RaceModel()
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
