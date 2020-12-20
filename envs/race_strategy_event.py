import math
import os
import sys
import csv
import re
from collections import deque, defaultdict
from os import listdir
from os.path import isfile, join
import random
from copy import deepcopy
import numpy as np
from gym import spaces
from gym import register
from sklearn.preprocessing import OneHotEncoder
from envs.planning_env import PlanningEnv
# Since the module race_simulation was forked it is not configured to work with the repository's folder structure


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_PATH, 'race_simulation'))

from race_simulation.racesim.src.race import Race
from race_simulation.racesim.src.import_pars import import_pars
from race_simulation.racesim.src.check_pars import check_pars
from race_simulation.racesim_basic.src.calc_racetimes_basic import calc_racetimes_basic

MAX_P = [1]
PROB_1 = [0.95, 0.05]
PROB_2 = [0.95, 0.025, 0.025]
PROB_3 = [0.91, 0.03, 0.03, 0.03]
PROBS = {1: MAX_P, 2: PROB_1, 3: PROB_2, 4:PROB_3}

MCS_PARS_FILE = 'pars_mcs.ini'

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


class RaceEnv(PlanningEnv):

    def __init__(self, gamma=0.95, horizon=20, scale_reward=True, positive_reward=True, start_lap=8,
                 verbose=False, config_path='./envs/race_strategy_model/active_drivers.csv', skip_steps=False,
                 n_cores=-1, randomize_events=False, rl_mode=False, log_path=None):

        super(RaceEnv, self).__init__()

        self._initials = {}
        self.verbose = verbose
        self.rl_mode = rl_mode
        self._actions_queue = deque()
        self._agents_queue = deque()
        self._last_pit = defaultdict(int)
        self.horizon = horizon
        self.gamma = gamma
        self.search_mode = False
        self.rollout_mode = False

        # TODO regulate dynamically the number of actions and drivers
        self.obs_dim = 30
        self.n_actions = 4
        self.action_space = spaces.Discrete(n=self.n_actions)
        self.observation_space = spaces.Box(low=0., high=self.horizon,
                                            shape=(self.obs_dim,), dtype=np.float64)
        self.scale_reward = scale_reward
        self.positive_reward = positive_reward
        self.viewer = None

        self._skip_steps = skip_steps

        self.sim_opts = {"use_prob_infl": True,
                    "create_rand_events": randomize_events,
                    "use_vse": False,
                    "no_sim_runs": 1,
                    "no_workers": 1,
                    "use_print": False,
                    "use_print_result": False,
                    "use_plot": False}
        
        if not log_path:
            # get repo path
            log_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # create output folders (if not existing)
        self.output_path = os.path.join(log_path, "sim_logs")

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

        self._tyre_expected_duration = None

        self._active_drivers_mapping = {}
        self._index_to_active = {}
        for i in range(len(self._active_drivers)):
            self._active_drivers_mapping[self._active_drivers[i]] = i
            self._index_to_active[i] = self._active_drivers[i]

        self.agents_number = len(self._active_drivers)

        self.start_lap = start_lap
        self._lap = 1
        self.race_length = 0

        self.presim_data = {}
        self._fcy_data = {}
        self.base_time = {}
        self.default_strategy = {}
        self.off_default = defaultdict(bool)
        self._t_pit = {}
        self.tireset_pars = {}
        self._strategies = defaultdict(list)
        self._pars_in = dict()

        self._t = -start_lap

        self._races_config_files = select_races(self._year)
        random.shuffle(self._races_config_files)

        self._compound_initials = []
        self._race_sim = None
        # TODO discard wet races

        self._drivers_number = 0

        # Take the base time, the predictions will be deltas from this time
        self.max_lap_time = 300
        self._drivers = []

        self._lap_time = None
        self._cumulative_time = None

        self._drivers_mapping = {}
        self._index_to_driver = {}

        self._pit_counts = defaultdict(int)
        self.used_compounds = [set() for _ in range(self.agents_number)]
        self._compound_initials = []
        self._available_compounds = {}
        self._compound_indices = {}

        self._flags_encoder = OneHotEncoder(sparse=False)
        self._compound_encoder = OneHotEncoder(sparse=False)
        self._flags_encoder.fit(np.array(FLAGS).reshape(-1, 1))
        self._compound_encoder.fit(np.array(COMPOUNDS).reshape(-1, 1))
        self.seed()

        if self.scale_reward and verbose:
            print("Reward is being normalized")

    def get_remaining_steps(self):
        """This is not exactly the timestep for the environment, but it is more meaningful when compared to the """
        horizon_delta = self.horizon - self._t
        race_delta = self.race_length - self._race_sim.get_cur_lap()
        return min(race_delta, horizon_delta)

    def get_max_ep_length(self):
        return min(self.get_race_length(), self.horizon)

    def get_race_length(self):
        assert self._race_sim is not None, "Race simulator not initialized yet"
        return self._race_sim.get_race_length()

    def get_state(self):
        if self.rl_mode:
            return self.__get_state_rl()
        else:
            return self.__get_complete_state()

    def __get_complete_state(self):
        state = self._race_sim.get_simulation_state()
        lap = state["lap"]
        current_lap_times = (state["lap_times"][lap] / self.max_lap_time).tolist()
        cumulative_times = (state["race_time"][lap] / 7200).tolist()  # A F1 race lasts at most 2h = 7200s
        flags = self._flags_encoder.transform(np.array([state['flag_state'][lap]]).reshape(-1, 1)).squeeze()
        overtake_ok = state['overtake_allowed'][lap].tolist()

        # lap /= self.race_length

        complete_state = []

        for d in state['drivers']:
            sim_index = self._race_sim.drivers_mapping[d.carno]
            available_compounds = np.zeros(len(COMPOUNDS)).T
            pit_count = 0
            if d.carno in self._active_drivers_mapping:
                available_tires = self._available_compounds[d.carno]
                for compound in available_tires:
                    if available_tires[compound] > 0:
                        available_compounds += self._compound_encoder.transform(
                            np.array([compound]).reshape(-1, 1)).squeeze()

                pit_count = self._pit_counts[d.carno] / 5

            overtake_available = overtake_ok[sim_index]
            current_tires = d.car.tireset.compound
            # current_tires = self._compound_encoder.transform(np.array([tires]).reshape(-1, 1)).squeeze()
            tire_age = d.car.tireset.age_tot / self.race_length
            lap_time = current_lap_times[sim_index]
            cumulative = cumulative_times[sim_index]

            changed_compound = len(self.used_compounds) > 1
            # flag = flags[sim_index] # TODO if flags are generated at runtime they must be included in the state

            driver_state = [lap_time, cumulative, pit_count, changed_compound, current_tires, tire_age, overtake_available]
            driver_state.extend(available_compounds)
            complete_state.append(driver_state)

        return complete_state

    def __get_state_rl(self):

        assert self.agents_number == 1, "This state getter is not correct for multi-agent settings"
        state = self._race_sim.get_simulation_state()
        assert self.race_length > 0, "Problem with race length"

        lap = state["lap"]
        current_lap_times = (state["lap_times"][lap] / self.max_lap_time).tolist()

        cumulative_times = (state["race_time"][lap] / 7200).tolist()  # A F1 race lasts at most 2h = 7200s
        # still_racing = state['still_racing'][lap].tolist()
        flags = self._flags_encoder.transform(np.array([state['flag_state'][lap]]).reshape(-1, 1)).squeeze()
        overtake_ok = state['overtake_allowed'][lap].tolist()
        # drs = state['drs']
        tires = []
        tire_age = []
        lap /= self.race_length

        for d in state['drivers']:
            if d.carno in self._active_drivers_mapping:
                sim_index = self._race_sim.drivers_mapping[d.carno]
                env_index = self._active_drivers_mapping[d.carno]
                available_tires = self._available_compounds[env_index]
                available_compounds = np.zeros(len(COMPOUNDS)).T
                for compound in available_tires:
                    if available_tires[compound] > 0:
                        available_compounds += self._compound_encoder.transform(np.array([compound]).reshape(-1, 1)).squeeze()

                overtake_available = overtake_ok[sim_index]
                tires = d.car.tireset.compound
                tires = self._compound_encoder.transform(np.array([tires]).reshape(-1, 1)).squeeze()
                tire_age = d.car.tireset.age_tot / self.race_length
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
               'last_pits': deepcopy(self._last_pit),
               #'simulator': deepcopy(self._race_sim.get_simulation_state()),
               'simulator': deepcopy(self._race_sim),
               't': deepcopy(self._t),
               'available_compounds': deepcopy(self._available_compounds),
               'pit_count': deepcopy(self._pit_counts),
               'used_compounds': deepcopy(self.used_compounds),
               'off_default': deepcopy(self.off_default),
               'strategies': deepcopy(self._strategies),
               'lap': deepcopy(self._lap)
               }
        return sig

    def set_signature(self, sig: dict) -> None:
        self.__set_state(sig["state"])
        self._race_sim = deepcopy(sig['simulator'])
        #self._race_sim.set_simulation_state(sig['simulator'])
        self._actions_queue = deepcopy(sig['action_queue'])
        self._agents_queue = deepcopy(sig['agents_queue'])
        self._last_pit = deepcopy(sig['last_pits'])
        self._t = deepcopy(sig['t'])
        self._available_compounds = deepcopy(sig['available_compounds'])
        self._pit_counts = deepcopy(sig['pit_count'])
        self.used_compounds = deepcopy(sig['used_compounds'])
        self.off_default = deepcopy(sig['off_default'])
        self._strategies = deepcopy(sig['strategies'])
        self._lap = deepcopy(sig['lap'])

    def get_distance_to_horizon(self) -> int:
        return self.race_length - self._race_sim.get_cur_lap()

    def get_available_actions(self, driver: int) -> list:
        """Allow another pit stop, signified by actions 1-3, only if the last pit
        for the same agents was at least 5 laps earlier"""

        actions = [0]
        # Check if the agent can do a pit stop
        if (self._last_pit[driver] > 5 and self._pit_counts[driver] < 2) \
                or (self._lap == self.race_length - 2 and len(self.used_compounds[driver]) == 1):
            # Check if the agent has left any of the tyres
            for i, compound in enumerate(self._compound_initials):
                if self._available_compounds[driver][compound] > 0:
                    actions.append(i+1)

        # Force pit-stop in penultimate lap if no two different compounds have been used or no pit stop has been done
        if self._lap == self.race_length - 2:
            if len(self.used_compounds[driver]) == 1:
                actions.remove(0)

            if len(self.used_compounds[driver]) == 1:
                used_compound = next(iter(self.used_compounds[driver])) # Avoid popping and adding back the item to set
                if self._available_compounds[driver][used_compound] > 0:
                    actions.remove(self._compound_indices[used_compound])

        return actions

    def enable_search_mode(self) -> None:
        """Activates default policy control for all non-controlled drivers"""

        self.search_mode = True
        previous_strategies = self._race_sim.set_controlled_drivers(self._drivers)

        # Count previous pit-stops and remove already used tyres from available tyres
        # to limit choice for non-controlled drivers
        for driver in previous_strategies:
            if driver not in self._active_drivers:
                for strategy in previous_strategies[driver]:
                    compound = strategy[1]
                    self._available_compounds[driver][compound] -= 1
                    self._pit_counts[driver] += 1
                # Remove one pit-stop because also the starting tyre is included in the previous list
                self._pit_counts[driver] -= 1
                self._last_pit[driver] = self._race_sim.get_cur_lap() - previous_strategies[driver][-1][0]

    def enable_rollout_mode(self) -> None:
        self.rollout_mode = True
        self._race_sim.set_enable_vse(True)
        self._race_sim.set_vse_enabled_drivers(self._active_drivers)

    def reset_stochasticity(self) -> None:
        """Regenerate random events in the simulator from this state onwards, as they need to be precomputed"""
        if self.sim_opts["create_rand_events"]:
            self._race_sim.handle_random_events_generation()

    def add_fcy_custom(self, fcy_type, stop=-1)->None:
        self._race_sim.handle_custom_fcy_generation(fcy_type, stop=stop)

    def map_action_to_compound(self, action_index: int) -> str:
        """Returns the compound name string for the desired input action"""
        assert len(self._compound_initials) > 0, "[ERROR] Env has not been reset yet"
        assert action_index <= len(self._compound_initials), "The desired compound is not enabled in this race"
        if action_index == 0:
            return ""
        else:
            return self._compound_initials[action_index - 1]

    def map_compound_to_action(self, compound_name: str) -> int:
        """Returns the action index corresponding to the input compound"""
        assert len(self._compound_initials) > 0, "[ERROR] Env has not been reset yet"
        assert compound_name in self._compound_initials, "The desired compound {} is not " \
                                                         "enabled in this race".format(compound_name)
        return self._compound_indices[compound_name]

    def partial_step(self, action, owner):
        """Accept an action for an agent, but perform the model transition only when an action for each agent has
        been specified"""

        agent = self.get_next_agent()
        self._agents_queue.popleft()
        assert owner == agent, "Actions are de-synchronized with agents queue {} {}".format(owner, agent)

        if not self.off_default[owner] and self.default_strategy[owner][self._lap] != action:
            self.off_default[owner] = True

        self._actions_queue.append((action, self._active_drivers_mapping[owner]))
        if action == 0:  # No pit, increment the time from last pit
            self._last_pit[owner] += 1
        else:  # Pit, reset time and remove a compound unit
            self._last_pit[owner] = 0
            # print("################")
            # print(self._pit_counts)
            # print(self._available_compounds)
            compound = self.map_action_to_compound(action)
            assert self._available_compounds[owner][compound] > 0, \
                "Trying to pit for a missing tyre unit"
            self._available_compounds[owner][compound] -= 1
            self.used_compounds[owner].add(compound)
            self._pit_counts[owner] += 1
            self._strategies[owner].append([self._lap, compound, 0, 0.])


        if len(self._actions_queue) == self.agents_number:  # We have an action for each agent, perform the transition
            actions = np.zeros(self.agents_number, dtype=int)
            while not len(self._actions_queue) == 0:
                action, owner_index = self._actions_queue.popleft()
                actions[owner_index] = action

            s, r, done, sig = self.__step(actions)
            # if action == 0:
            #
            #     print(r)
            return s,r,done, sig

        else:  # No transition, we don't have actions for all the agents
            print("AAA")
            return self.get_state(), np.zeros(self.agents_number), self.is_terminal(), {}

    def has_transitioned(self) -> bool:
        """Check if the environment has performed a state transition"""
        return len(self._actions_queue) == 0

    def step(self, action: int):
        """
        Step wrapper for single-agent execution

        :param action: the index of the action to be executed
        :type action: int
        """

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

        if not self.search_mode and self._lap + 2 in self._fcy_data:
            end_lap, fcy_type = self._fcy_data[self._lap + 2]
            self.add_fcy_custom(fcy_type, stop=end_lap)

        if self.is_terminal():
            print("BBB", self._t, self.horizon, self._lap, self._race_sim.get_race_length())
            return self.get_state(), np.zeros(self.agents_number), True, {}

        pit_info = []
        for action, idx in zip(actions, range(len(actions))):
            if action > 0:
                compound = self.map_action_to_compound(action)
                pit_info.append((self._index_to_active[idx], [compound, 0, 0.]))

        # Use default strategies for all non-controlled drivers during search in planners
        if self.search_mode:
            for d in self._drivers:
                if d not in self._active_drivers_mapping:
                    actions = self.get_default_strategy(d) # get_available_actions(d)
                    prob = PROBS[len(actions)]
                    a = np.random.choice(actions, p=prob)
                    execute = np.random.random() > 0.9 # Drivers have a 10% probability of postponing the pit
                    if a > 0 and execute:
                        compound = self.map_action_to_compound(a)
                        self._available_compounds[d][compound] -= 1
                        self.used_compounds[d].add(compound)
                        self._pit_counts[d] += 1
                        pit_info.append((d, [compound, 0, 0.]))

        predicted_times, driver_info = self._race_sim.step(pit_info)

        lap_times = np.ones(self._drivers_number) * self.max_lap_time
        for lap_time, driver in zip(predicted_times, driver_info):
            index = self._drivers_mapping[driver.carno]
            lap_times[index] = lap_time
            # self._cumulative_time[index] += lap_time

        self._t += 1
        self._lap = self._race_sim.get_cur_lap()

        reward = np.ones(self.agents_number) * self.max_lap_time
        for driver in self._active_drivers:
            active_index = self._active_drivers_mapping[driver]
            index = self._drivers_mapping[driver]
            reward[active_index] = -np.clip(lap_times[index], 0, self.max_lap_time)

        if self.scale_reward:
            reward /= self.max_lap_time
            if self.positive_reward:
                reward = 1 + reward

        # print(self._lap, self._race_length, self._terminal)

        # if self._terminal:
        #     print("//////////////////////////////////", reward)

        return self.get_state(), reward, self.is_terminal(), {}

    def save_results(self, timestamp):
        #_path = self.results_path + timestamp
        save_path = os.path.join(self.results_path, timestamp)
        os.makedirs(save_path, exist_ok=True)
        self._race_sim.export_results_as_csv(results_path=save_path)

    def simulate_strategy(self, pars_in: dict, initials: str, strategy:list):
        team = pars_in['driver_pars'][initials]['team']
        car_pars = pars_in['car_pars'][team]
        driver_pars = pars_in['driver_pars'][initials]
        track_pars = pars_in['track_pars']
        driver_no = driver_pars['carno']

        t_quali = pars_in['track_pars']['t_q']
        race_gap = pars_in['track_pars']['t_gap_racepace']
        t_driver = driver_pars['t_driver']
        t_car = car_pars['t_car']
        t_base = t_quali + race_gap + t_car + t_driver + pars_in['track_pars']['t_lap_sens_mass'] * car_pars['m_fuel']
        self.base_time[driver_no] = t_base
        self.default_strategy[driver_no] = defaultdict(int)
        self.tireset_pars[driver_no] = pars_in['tireset_pars'][initials]

        # Store the default strategy
        for pit in driver_pars['strategy_info']:
            if pit[0] > 0:
                self.default_strategy[driver_no][pit[0]] = self.map_compound_to_action(pit[1])

        t_pit_tirechange_min = pars_in['track_pars']['t_pit_tirechange_min']
        t_pit_tirechange_add = pars_in['car_pars'][team]['t_pit_tirechange_add']
        t_pit_tirechange = t_pit_tirechange_min + t_pit_tirechange_add
        self._t_pit[driver_no] = t_pit_tirechange + track_pars["t_pitdrive_inlap"] + track_pars["t_pitdrive_outlap"]

        # Compute the lap times for the default strategy, ignoring any interference effects
        res = calc_racetimes_basic(t_base=t_base,
                                    tot_no_laps=pars_in['race_pars']["tot_no_laps"],
                                    t_lap_sens_mass=track_pars["t_lap_sens_mass"],
                                    t_pitdrive_inlap=track_pars["t_pitdrive_inlap"],
                                    t_pitdrive_outlap=track_pars["t_pitdrive_outlap"],
                                    t_pitdrive_inlap_fcy=track_pars["t_pitdrive_inlap_fcy"],
                                    t_pitdrive_outlap_fcy=track_pars["t_pitdrive_outlap_fcy"],
                                    t_pitdrive_inlap_sc=track_pars["t_pitdrive_inlap_sc"],
                                    t_pitdrive_outlap_sc=track_pars["t_pitdrive_outlap_sc"],
                                    pits_aft_finishline=track_pars["pits_aft_finishline"],
                                    t_pit_tirechange=t_pit_tirechange,
                                    tire_pars=pars_in['tireset_pars'][initials],
                                    p_grid=driver_pars["p_grid"],
                                    t_loss_pergridpos=track_pars["t_loss_pergridpos"],
                                    t_loss_firstlap=track_pars["t_loss_firstlap"],
                                    strategy=strategy,
                                    drivetype=car_pars["drivetype"],
                                    m_fuel_init=car_pars["m_fuel"],
                                    b_fuel_perlap=car_pars["b_fuel_perlap"],
                                    t_pit_refuel_perkg=car_pars["t_pit_refuel_perkg"],
                                    t_pit_charge_perkwh=car_pars["t_pit_charge_perkwh"],
                                    fcy_phases=None,
                                    t_lap_sc=track_pars["mult_t_lap_sc"] * t_base,
                                    t_lap_fcy=track_pars["mult_t_lap_fcy"] * t_base,
                                    deact_pitstop_warn=True)
        return res[0]

    def reset(self, quantile_strategies=False):

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

        self.search_mode = False
        race_pars_file = self._races_config_files.pop(0)
        # print(race_pars_file)
        self._races_config_files.append(race_pars_file)

        # race_pars_file = "pars_Melbourne_2017.ini"
        # race_pars_file = "pars_SaoPaulo_2018.ini"
        # race_pars_file = "pars_Suzuka_2015.ini"
        # race_pars_file = "pars_Suzuka_2016.ini"
        # race_pars_file = "pars_SaoPaulo_2018.ini"
        # race_pars_file = "pars_Spielberg_2017.ini"
        race_pars_file = "pars_Shanghai_2018.ini"


        # load parameters
        pars_in, vse_paths = import_pars(use_print=self.sim_opts["use_print"],
                                         use_vse=self.sim_opts["use_vse"],
                                         race_pars_file=race_pars_file,
                                         mcs_pars_file=MCS_PARS_FILE)

        # check parameters
        check_pars(sim_opts=self.sim_opts, pars_in=pars_in)

        self._race_sim = Race(race_pars=pars_in["race_pars"],
                              driver_pars=pars_in["driver_pars"],
                              car_pars=pars_in["car_pars"],
                              tireset_pars=pars_in["tireset_pars"],
                              track_pars=pars_in["track_pars"],
                              vse_pars=pars_in["vse_pars"],
                              vse_paths=vse_paths,
                              use_prob_infl=self.sim_opts['use_prob_infl'],
                              create_rand_events=self.sim_opts['create_rand_events'],
                              monte_carlo_pars=pars_in["monte_carlo_pars"],
                              event_pars=pars_in["event_pars"],
                              disable_retirements=True)

        self._race_sim.set_enable_vse(False)

        self._compound_initials = pars_in["vse_pars"]["param_dry_compounds"]
        state = self._race_sim.get_simulation_state()

        self._fcy_data = {}
        for event in pars_in["event_pars"]["fcy_data"]["phases"]:
            start_lap = float(event[0])
            end_lap = float(event[1])
            fcy_type = event[2]
            print(start_lap, end_lap, fcy_type)
            self._fcy_data[math.floor(start_lap)] = (end_lap, fcy_type)

        # Compute the expected duration for each tire compound

        self.race_length = self._race_sim.get_race_length()

        self._tyre_expected_duration={}

        if quantile_strategies:  # Use an "a posteriori" approach
            stints = defaultdict(list)

            for driver in state["drivers"]:
                strategy = driver.strategy_info

                if len(strategy) > 1: # If the strategy is only one stint long, the driver has crashed
                    for i, stint_info in enumerate(strategy):
                     if i < len(strategy) - 1 :
                         next_stint = strategy[i+1]
                         stint_duration = next_stint[0] - stint_info[0]
                     else:
                         stint_duration = self.race_length - stint_info[0]
                     stints[stint_info[1]].append(stint_duration)

            for compound in stints:
                self._tyre_expected_duration[compound] = np.quantile(stints[compound], 0.6)
            # print("[DEBUG] --> " + str(pars_in['track_pars']))
        else:
            self._tyre_expected_duration = pars_in['track_pars']['predicted_duration']

        # Store the mapping between action indices and compounds
        for i, compound in enumerate(self._compound_initials):
            self._compound_indices[compound] = i+1

        # Initialize the driver number / array indices mapping
        self._drivers = [driver.carno for driver in state["drivers"]]
        self._drivers_number = len(self._drivers)
        for i in range(self._drivers_number):
            self._drivers_mapping[self._drivers[i]] = i
            self._index_to_driver[i] = self._drivers[i]

        # Use the default strategies before the actual start
        while self._race_sim.get_cur_lap() < self.start_lap:
            self._race_sim.step([])
        self._t = 0
        start_strategies = self._race_sim.set_controlled_drivers(list(self._active_drivers))

        # Setup initial tyre allocation for different availabilities
        if len(self._compound_initials) == 2:
            default_availabilities = {self._compound_initials[0]: 2, self._compound_initials[1]: 3}
        elif len(self._compound_initials) == 3:
            default_availabilities = {self._compound_initials[0]: 1,
                                      self._compound_initials[1]: 2,
                                      self._compound_initials[2]: 2}
        else:
            raise ValueError("Unexpected compound availabilities number, {}", self._compound_initials)
        self._available_compounds = {driver : deepcopy(default_availabilities) for driver in self._drivers}

        self.used_compounds = {driver: set() for driver in self._drivers}
        self._strategies = defaultdict(list)
        # Enable pit stops for the drivers and remove one unit the starting tire from the available compounds
        for d in self._active_drivers:
            self._last_pit[d] = 6
            start_compound = start_strategies[d][0][1]
            self._strategies[d].append(start_strategies[d][0])
            self.used_compounds[d].add(start_compound)
            self._available_compounds[d][start_compound] -= 1

        # Reset the pit counter
        self._pit_counts = defaultdict(int)

        self._t_pit = {}
        self.base_time = {}
        self.default_strategy = {}
        self.off_default = defaultdict(bool)
        self.tireset_pars = {}
        self._pars_in = pars_in
        self._initials = {}

        # Run a reduced simulation for the strategy outcome and set default values for time losses
        for driver in self._active_drivers:
            index = self._drivers_mapping[driver]
            initials = state['drivers'][index].initials
            strategy = pars_in['driver_pars'][initials]['strategy_info']
            self.presim_data[driver] = self.simulate_strategy(pars_in=pars_in, initials=initials, strategy=strategy)
            self._initials[driver] = initials

        self._lap = self._race_sim.get_cur_lap()

        return self.get_state()

    def get_agents_standings(self):
        temp = np.argsort(self._cumulative_time)
        ranks = deque()
        for index in temp:
            driver = self._index_to_driver[index]
            if driver in self._active_drivers_mapping:
                ranks.append(self._active_drivers_mapping[driver])
        return ranks

    def get_next_agent(self):
        if len(self._agents_queue) == 0:
            for car_number in self._active_drivers:
                self._agents_queue.append(car_number)
            # self._agents_queue = self.get_agents_standings()
        return self._agents_queue[0]

    def is_terminal(self):
        return self._lap >= self.race_length or self._t >= self.horizon

    def get_remaining_compounds(self, driver) -> list:
        remaining = []
        for compound in self._available_compounds[driver]:
            if self._available_compounds[driver][compound] > 0:
                remaining.append(compound)
        return remaining

    def get_default_strategy(self, owner):
        assert self._tyre_expected_duration is not None, "You need to reset the environment before " \
                                                   "you can get the default tyre durations"
        compound, age = self._race_sim.get_tyre_age(owner)
        remaining = self.get_remaining_compounds(owner)
        if age >= self._tyre_expected_duration[compound] and len(remaining) > 0 and self._last_pit[owner] > 5:
            missing_laps = self.race_length - self._race_sim.get_cur_lap()

            remaining = np.array(remaining)
            delta = np.array([self._tyre_expected_duration[compound] - missing_laps for compound in remaining])
            positive = np.argwhere(delta >= 0)

            # If any compound allows to complete the race, use the one that fits better the remaining laps
            if len(positive) > 0:
                remaining = remaining[positive]
                delta = delta[positive]
                remaining = remaining.ravel()
                delta = delta.ravel()
                # Force change of compound to avoid penalty at the end of the race
                if len(self.used_compounds[owner]) == 1:
                    start_compound = next(iter(self.used_compounds[owner]))
                    allowed = np.argwhere(remaining != start_compound)
                    remaining = remaining[allowed]
                    delta = delta[allowed]
                assert len(remaining) > 0, "Something wrong with tyre change constraint"

            remaining = remaining.tolist()
            delta = delta.tolist()
            best = remaining[np.argmin(delta)]
            if type(best) == list:
                best = best[0]
            return [self.map_compound_to_action(best)]
        else:
            return [0]

    def get_mean_estimation(self, action, owner):
        """Returns an estimation for the value of the current action in the race,
        without considering stochastic effects"""
        if self.is_terminal():
            return 0, True
        if not self.off_default[owner] and self.default_strategy[owner][self._lap] == action:
            return self.presim_data[owner][self._lap], True
        else:
            strategy = deepcopy(self._strategies[owner])
            if action > 0:
                compound = self.map_action_to_compound(action)
                strategy.append([self._lap, compound, 0, 0.0])
            initials = self._initials[owner]
            race_time = self.simulate_strategy(self._pars_in, initials, strategy)[self._lap]
            return race_time, False


register(
    id='RaceStrategy-v2',
    entry_point='envs.race_strategy_full:RaceModel'
)
