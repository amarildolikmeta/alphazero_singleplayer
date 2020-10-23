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

# Since the module race_simulation was forked it is not configured to work with the repository's folder structure
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_PATH, 'race_simulation'))

from race_simulation.racesim.src.race import Race
from race_simulation.racesim.src.import_pars import import_pars
from race_simulation.racesim.src.check_pars import check_pars

MCS_PARS_FILE = 'pars_mcs.ini'

SIM_OPTS = {"use_prob_infl": True,
            "create_rand_events": False,
            "use_vse": False,
            "no_sim_runs": 1,
            "no_workers": 1,
            "use_print": False,
            "use_print_result": False,
            "use_plot": False}

COMPOUND_MAPPING = {"A1": 1,
                    "A2": 2,
                    "A3": 3,
                    "A4": 4,
                    "A5": 5,
                    "A6": 6,
                    "I": 7,
                    "W": 8}

FLAGS = ["G", "Y", "R", "FCY", "SC", "VSC"]
COMPOUNDS = ["A1", "A2", "A3", "A4", "A5", "A6", "I", "W"]


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

    def __init__(self, gamma=0.95, horizon=20, scale_reward=True, positive_reward=True, start_lap=8,
                 verbose=False, config_path='./envs/race_strategy_model/active_drivers.csv', skip_steps=False, n_cores=-1):
        # print("////////////////////////////////////////", horizon)
        self.verbose = verbose
        self._actions_queue = deque()
        self._agents_queue = deque()
        self._agents_last_pit = defaultdict(int)
        self.horizon = horizon
        self.gamma = gamma
        # TODO regulate dynamically the number of actions and drivers
        self.obs_dim = 30
        self.n_actions = 3
        self.action_space = spaces.Discrete(n=self.n_actions)
        self.observation_space = spaces.Box(low=0., high=self.horizon,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.scale_reward = scale_reward
        self.positive_reward = positive_reward
        self.viewer = None

        self._skip_steps = skip_steps
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
        self.max_lap_time = 300
        self._drivers = []
        self._race_length = 0

        self._lap_time = None
        self._cumulative_time = None

        self._drivers_mapping = {}
        self._index_to_driver = {}

        self._terminal = False
        self._pit_counts = [0] * self.agents_number
        self.used_compounds = [set() for _ in range(self.agents_number)]
        self._compound_initials = []
        self._available_compounds = []

        self._flags_encoder = OneHotEncoder(sparse=False)
        self._compound_encoder = OneHotEncoder(sparse=False)
        self._flags_encoder.fit(np.array(FLAGS).reshape(-1, 1))
        self._compound_encoder.fit(np.array(COMPOUNDS).reshape(-1, 1))
        self.seed()

        if self.scale_reward and verbose:
            print("Reward is being normalized")

    def get_race_length(self):
        assert self._race_sim is not None, "Race simulator not initialized yet"
        return self._race_sim.get_race_length()

    def get_state(self, controlled_only=False):
        """Get a state representation suitable for a RL agent"""

        if self.agents_number > 1:
            return None
        state = self._race_sim.get_simulation_state()
        assert self.race_length > 0, "Problem with race length"

        lap = state[0]
        current_lap_times = (state[1][lap] / self.max_lap_time).tolist()

        cumulative_times = (state[2][lap] / 7200).tolist()  # A F1 race lasts at most 2h = 7200s
        # still_racing = state['still_racing'][lap].tolist()
        flags = self._flags_encoder.transform(np.array([state[4][lap]]).reshape(-1, 1)).squeeze()
        overtake_ok = state[5][lap].tolist()
        # drs = state['drs']
        tires = []
        tire_age = []
        lap /= self.race_length

        for d in state[10]:
            carno = d[2]
            if carno in self._active_drivers:
                sim_index = self._race_sim.drivers_mapping[carno]
                env_index = self._active_drivers_mapping[carno]
                available_tires = self._available_compounds[env_index]
                available_compounds = np.zeros(len(COMPOUNDS)).T
                for compound in available_tires:
                    if available_tires[compound] > 0:
                        available_compounds += self._compound_encoder.transform(np.array([compound]).reshape(-1, 1)).squeeze()

                overtake_available = overtake_ok[sim_index]
                car = d[1]
                tireset = car[1]
                tires = tireset[0]
                age_tot = tireset[1]
                tires = self._compound_encoder.transform(np.array([tires]).reshape(-1, 1)).squeeze()
                tire_age = age_tot / self.race_length
                lap_time = current_lap_times[sim_index]
                cumulative = cumulative_times[sim_index]
                pit_count = self._pit_counts[env_index] / 5
                changed_compound = len(self.used_compounds) > 1
                break


        # Build the state matrix with shape (features, drivers)
        rl_state = [lap, lap_time, cumulative, tire_age, pit_count, changed_compound,
                    overtake_available]
        rl_state = np.asarray(rl_state, dtype=float).T
        rl_state = np.hstack((rl_state, tires, available_compounds))

        # Make a single row from the matrix
        rl_state = rl_state.ravel()
        rl_state = np.hstack(([lap], flags, rl_state))
        # print(rl_state.shape)

        return rl_state

    def __set_state(self, state):
        pass

    def get_signature(self) -> dict:
        sig = {'state': deepcopy(self.get_state()),
               'action_queue': deepcopy(self._actions_queue),
               'agents_queue': deepcopy(self._agents_queue),
               'last_pits': deepcopy(self._agents_last_pit),
               'simulator_state': self._race_sim.get_simulation_state(),
               't': deepcopy(self._t),
               'terminal': deepcopy(self._terminal),
               'available_compounds': deepcopy(self._available_compounds),
               'pit_count': deepcopy(self._pit_counts),
               'used_compounds': deepcopy(self.used_compounds)
               }
        return sig

    def set_signature(self, sig: dict) -> None:
        self.__set_state(sig["state"])
        self._race_sim.set_simulation_state(sig['simulator_state'])
        self._actions_queue = deepcopy(sig['action_queue'])
        self._agents_queue = deepcopy(sig['agents_queue'])
        self._agents_last_pit = deepcopy(sig['last_pits'])
        self._t = deepcopy(sig['t'])
        self._terminal = deepcopy(sig['terminal'])
        self._available_compounds = deepcopy(sig['available_compounds'])
        self._pit_counts = deepcopy(sig['pit_count'])
        self.used_compounds = deepcopy(sig['used_compounds'])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_available_actions(self, agent: int) -> list:
        """Allow another pit stop, signified by actions 1-3, only if the last pit
        for the same agents was at least 5 laps earlier"""

        actions = [0]
        # Check if the agent can do a pit stop
        if self._agents_last_pit[agent] > 5:
            # Check if the agent has left any of the tyres
            for i in range(1, len(self._compound_initials) + 1):
                if self._available_compounds[agent][self.map_action_to_compound(i)] > 0:
                    actions.append(i)
        return actions

    def map_action_to_compound(self, action_index: int) -> str:
        assert len(self._compound_initials) > 0, "[ERROR] Env has not been reset yet"
        assert action_index <= len(self._compound_initials), "The desired compound is not enabled in this race"
        if action_index == 0:
            return ""
        else:
            return self._compound_initials[action_index - 1]

    def partial_step(self, action, owner):
        """Accept an action for an agent, but perform the model transition only when an action for each agent has
        been specified"""

        agent = self.get_next_agent()
        self._agents_queue.popleft()
        assert owner == agent, "Actions are de-synchronized with agents queue {} {}".format(owner, agent)

        self._actions_queue.append((action, owner))
        if action == 0:  # No pit, increment the time from last pit
            self._agents_last_pit[owner] += 1
        else:  # Pit, reset time and remove a compound unit
            self._agents_last_pit[owner] = 0
            # print("################")
            # print(self._pit_counts)
            # print(self._available_compounds)
            compound = self.map_action_to_compound(action)
            assert self._available_compounds[owner][compound] > 0, \
                "Trying to pit for a missing tyre unit"
            self._available_compounds[owner][compound] -= 1
            self.used_compounds[owner].add(compound)
            self._pit_counts[owner] += 1

        if len(self._actions_queue) == self.agents_number:  # We have an action for each agent, perform the transition
            actions = np.zeros(self.agents_number, dtype=int)
            while not len(self._actions_queue) == 0:
                action, owner_index = self._actions_queue.popleft()
                actions[owner_index] = action
            return self.__step(actions)

        else:  # No transition, we don't have actions for all the agents
            print("AAA")
            return self.get_state(), np.zeros(self.agents_number), self._terminal, {}

    def has_transitioned(self):
        return len(self._actions_queue) == 0

    def step(self, action):
        """Step wrapper for single-agent execution"""

        if type(action) == np.ndarray and len(action) == 1:
            action = action[0]
        elif type(action) == np.int64:
            action = int(action)
        elif type(action) == int:
            pass
        else:
            raise NotImplementedError

        assert self.agents_number == 1, "Cannot use this function with more than one agent, use partial_step instead"
        agent = self.get_next_agent()

        # Fix actions for RL agents with a fixed action space
        if self._skip_steps and not(action in self.get_available_actions(agent)):
            action = 0

        state, rew, terminal, sig = self.partial_step(action, agent)

        # No need to do step by step for an RL agent if there is only one action available, fast forward the env
        # to the next decision step
        if self._skip_steps:
            while len(self.get_available_actions(agent)) == 1 and not terminal:
                state, reward, terminal, sig = self.partial_step(0, agent)
                rew += reward

        return state, rew[0], terminal, sig

    def __step(self, actions: np.ndarray):
        assert self._race_sim is not None, "[ERROR] Tried to perform a step in the environment before resetting it"

        self._lap = self._race_sim.get_cur_lap()
        self._terminal = True if self._t >= self.horizon or self._lap >= self._race_sim.get_race_length() else False
        if self._terminal:
            print("BBB", self._t, self.horizon, self._lap, self._race_sim.get_race_length())
            return self.get_state(), np.zeros(self.agents_number), True, {}

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
            # self._cumulative_time[index] += lap_time

        self._t += 1
        self._lap = self._race_sim.get_cur_lap()
        self._terminal = True if self._t >= self.horizon or self._lap >= self._race_sim.get_race_length() else False

        reward = np.ones(self.agents_number) * self.max_lap_time
        for driver in self._active_drivers:
            active_index = self._active_drivers_mapping[driver]
            index = self._drivers_mapping[driver]
            reward[active_index] = -np.clip(lap_times[index], 0, self.max_lap_time)

        if self._terminal:  # Penalize if no pit stop has been done or if no two different compounds have been used
            for i in range(self.agents_number):
                reward[i] = -10000 if self._pit_counts == 0 or len(self.used_compounds) == 1 else reward[i]

        if self.scale_reward:
            reward /= self.max_lap_time
            if self.positive_reward:
                reward = 1 + reward

        # print(self._lap, self._race_length, self._terminal)

        # if self._terminal:
        #     print("//////////////////////////////////", reward)

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
        self._terminal = False
        race_pars_file = self._races_config_files.pop(0)
        # print(race_pars_file)
        self._races_config_files.append(race_pars_file)

        race_pars_file = "pars_Catalunya_2017.ini"

        # load parameters
        pars_in, vse_paths = import_pars(use_print=SIM_OPTS["use_print"],
                                         use_vse=SIM_OPTS["use_vse"],
                                         race_pars_file=race_pars_file,
                                         mcs_pars_file=MCS_PARS_FILE)

        # check parameters
        check_pars(sim_opts=SIM_OPTS, pars_in=pars_in)

        self._race_sim = Race(race_pars=pars_in["race_pars"],
                              driver_pars=pars_in["driver_pars"],
                              car_pars=pars_in["car_pars"],
                              tireset_pars=pars_in["tireset_pars"],
                              track_pars=pars_in["track_pars"],
                              vse_pars=pars_in["vse_pars"],
                              vse_paths=vse_paths,
                              use_prob_infl=SIM_OPTS['use_prob_infl'],
                              create_rand_events=SIM_OPTS['create_rand_events'],
                              monte_carlo_pars=pars_in["monte_carlo_pars"],
                              event_pars=pars_in["event_pars"],
                              disable_retirements=True)

        # Remove the VSE because it contains Tensorflow objects which can't be serialized, its work is already done
        # during the initialization of the Race object
        self._race_sim.vse = None

        self._compound_initials = pars_in["vse_pars"]["param_dry_compounds"]
        state = self._race_sim.get_simulation_state()

        # Initialize the driver number / array indices mapping
        self._drivers = [driver[2] for driver in state[10]]
        self._drivers_number = len(self._drivers)
        for i in range(self._drivers_number):
            self._drivers_mapping[self._drivers[i]] = i
            self._index_to_driver[i] = self._drivers[i]

        # Use the default strategies before the actual start
        while self._race_sim.get_cur_lap() < self.start_lap:
            self._race_sim.step([])
        self._t = 0
        start_strategies = self._race_sim.set_controlled_drivers(list(self._active_drivers))
        self.race_length = self._race_sim.get_race_length()

        # Setup initial tyre allocation for different availabilities
        if len(self._compound_initials) == 2:
            default_availabilities = {self._compound_initials[0]: 3, self._compound_initials[1]: 2}
        elif len(self._compound_initials) == 3:
            default_availabilities = {self._compound_initials[0]: 2,
                                      self._compound_initials[1]: 2,
                                      self._compound_initials[2]: 1}
        else:
            raise ValueError("Unexpected compound availabilities number, {}", self._compound_initials)
        self._available_compounds = [deepcopy(default_availabilities)] * len(self._active_drivers)

        self.used_compounds = [set() for _ in range(self.agents_number)]
        # Enable pit stops for the drivers and remove one unit the starting tire from the available compounds
        for i in range(self.agents_number):
            self._agents_last_pit[i] = 6
            driver = self._index_to_active[i]
            start_compound = start_strategies[driver]
            self.used_compounds[i].add(start_compound)
            self._available_compounds[i][start_compound] -= 1

        # Reset the pit counter
        self._pit_counts = [0] * self.agents_number

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
