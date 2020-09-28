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


def generate_race_full(**game_params):
    if game_params is None:
        game_params = {}
    return RaceModel(**game_params)


def get_current_tyres(df: pd.DataFrame):
    df = df.squeeze()
    if df['tyre_1'] == 1:
        return 'tyre_1'
    elif df['tyre_2'] == 1:
        return 'tyre_2'
    elif df['tyre_3'] == 1:
        return 'tyre_3'
    elif df['tyre_4'] == 1:
        return 'tyre_4'
    elif df['tyre_5'] == 1:
        return 'tyre_5'
    elif df['tyre_6'] == 1:
        return 'tyre_6'


def find_available_rubber(race):
    available = []
    for col in ['tyre_1', 'tyre_2', 'tyre_3', 'tyre_4', 'tyre_5', 'tyre_6']:
        if race[col].any() > 0:
            available.append(col)

    if len(available) < 3:
        assert len(available) == 2, \
            "Only one available tyre for race {}, check the dataset".format(race['raceId'].values()[0])
        min_avail = int(available[0][5])
        max_avail = int(available[1][5])

        if max_avail - min_avail == 2:
            missing = "tyre_" + str(min_avail + 1)
            available = [available[0], missing, available[1]]
        else:
            missing = "tyre_" + str(max_avail + 1)
            available.append(missing)
    return available[0], available[1], available[2]

def compute_ranking(cumulative_time):
    """Computes the ranking on track, based on cumulative times"""
    temp = np.argsort(cumulative_time)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(cumulative_time)) + 1
    return ranks


def get_default_strategies(race):
    soft, med, hard = find_available_rubber(race)
    tyre_stints = {soft: [], med: [], hard: []}

    for driver in race['driverId'].unique():
        driver_laps = race[race['driverId'] == driver].sort_values('lap')
        for compound in [soft, med, hard]:
            laps_with_compound = driver_laps[driver_laps[compound] == 1]
            if laps_with_compound['lap'].count() > 0:
                prev_lap = -1
                stint_length = 0
                for index, row in laps_with_compound.sort_values('lap').iterrows():
                    if row['unnorm_lap'] - prev_lap > 1 and not (prev_lap == -1):
                        tyre_stints[compound].append(stint_length)
                        stint_length = 1
                    else:
                        stint_length += 1
                    prev_lap = row['lap']

    for compound in [soft, med, hard]:
        if len(tyre_stints[compound]) > 0:
            tyre_stints[compound] = (np.mean(tyre_stints[compound]), np.std(tyre_stints[compound]))
        else:
            tyre_stints[compound] = (np.inf, np.inf)

    return tyre_stints


def compute_state(row):
    if row['pit'].all() != 0:
        return 'pit'
    elif row['safety'].all() != 0:
        return 'safety'
    else:
        return 'regular'


class RaceModel(gym.Env):

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

        self._model = RaceStrategyModel(year=self._year, verbose=False, n_cores=n_cores)
        self._model.train() #

        self._drivers_number = 0

        # Take the base time, the predictions will be deltas from this time
        self._base_time = 0
        self.max_lap_time = 0
        self._drivers = None
        self._race_length = 0
        self._default_strategies = None
        self._laps = None

        self._pit_states = None
        self._pit_counts = None
        self._pit_costs = None
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
        #start = time.time()
        state = [deepcopy(self._lap),
                 deepcopy(self._current_tyres),
                 deepcopy(self._cumulative_time),
                 deepcopy(self._lap_time),
                 deepcopy(self._pit_states),
                 deepcopy(self._pit_counts),
                 deepcopy(self._tyre_age)]
        #stop = time.time()
        #print("get state time:", stop - start)
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
            self._tyre_age[index] = 0

        elif self._pit_states[index] == 1:
            self._pit_states[index] = 0

        else:
            self._pit_costs[index] = 0

    def get_available_actions(self, agent: int):
        """Allow another pit stop, signified by actions 1-3, only if the last pit
        for the same agents was at least 5 laps earlier"""
        if self._agents_last_pit[agent] > 5:
            return [0,1,2,3]
        else:
            return [0]

    def partial_step(self, action, owner):
        """Accept an action for an gent, but perform the model transition only when an action for each agent has
        been specified"""

        agent = self.get_next_agent()
        self._agents_queue.popleft()
        assert owner == agent, "Actions are de-synchronized with agents queue {} {}".format(owner, agent)

        self._actions_queue.append((action, owner))
        if action == 0: # No pit, increment the time from last pit
            self._agents_last_pit[owner] += 1
        else: # Pit, reset time
            self._agents_last_pit[owner] = 0

        if len(self._actions_queue) == self.agents_number:
            actions = np.zeros(self.agents_number)
            while not len(self._actions_queue) == 0:
                action, owner_index = self._actions_queue.popleft()
                actions[owner_index] = action
            return self.step(actions)

        else:
            return self.get_state(), self._reward, self._terminal, {}

    def has_transitioned(self):
        return len(self._actions_queue) == 0

    def step(self, actions: np.ndarray):
        #start = time.time()
        # assert len(actions) == len(self._active_drivers), \
        #     "Fewer actions were provided than the number of active drivers"
        reward = np.zeros(len(actions))
        lap = self.start_lap + self._t + 1

        if self._t >= self.horizon or lap > len(self._laps):
            return self.get_state(), [0] * self.agents_number, True, {}

        self._lap = lap
        lap_norm = self._laps[lap - 1]

        # Update current lap times and compute delta wrt previous lap
        self._old_lap_time = self._lap_time.copy()
        self._lap_time = self._next_lap_time.copy()
        self._lap_deltas = self._lap_time - self._old_lap_time
        self._cumulative_time = self._cumulative_time + self._lap_time

        # Compute current positions for XGBoost predicted cumulative times
        ranks = compute_ranking(self._cumulative_time)

        safety_laps = self._model.test_race[self._model.test_race['lap'] == lap_norm]['safety'].max()

        x_data = []
        x_pit_data = []
        no_pit = []
        pitting = []

        # race_laps = self._model.test_race
        laps_db = self._model.laps_database
        #stop = time.time()

        #print("Step init time:", stop - start)

        #contribute = defaultdict(int)

        started = lap > self.start_lap

        # Simulate each driver
        for driver in self._drivers_mapping:
            #start = time.time()
            index = self._drivers_mapping[driver]

            # Take the lap for the current driver

            row = laps_db[(driver, lap_norm)]
            #row = race_laps[(race_laps['driverId'] == driver) & (race_laps['lap'] == lap_norm)].sort_values('lap')

            # The driver might have retired if no row is available in the dataset
            is_out = False
            if row is None:
                is_out = True
                row = self._last_available_row[index]
            else:
                self._last_available_row[index] = row.copy()

            # Just copy the dataset information if we are below the simulation start lap
            if lap < self.start_lap and not is_out:
                self._lap_time[index] = row.squeeze()['milliseconds']

            #stop = time.time()
            #contribute['fetch'] += stop - start


            #start = time.time()
            # Find the index of the drivers following and preceding the current car
            position = ranks[index]

            in_front = -1
            following = -1

            if position > 1:  # The leader has no car in front, consider only other cars
                in_front = np.argwhere(ranks == position - 1)

            follower = np.argwhere(ranks == position + 1)
            if position < self._drivers_number and not (self._cumulative_time[follower] == np.inf):  # The last car has no follower
                following = follower

            # Fill the template dataframe with the features extracted from predicted time
            #start = time.time()
            data = row.copy()
            #stop = time.time()

            #contribute['copy'] += stop - start

            data['milliseconds'] = self._lap_time[index].squeeze()
            data['cumulative'] = self._cumulative_time[index].squeeze()
            data['position'] = position
            data['drs'] = 0
            data['battle'] = 1

            # Compute gaps and features to car in front, if any
            if in_front > -1:
                data['delta_car_in_front'] = (self._lap_time[index] - self._lap_time[in_front]).squeeze()
                data['time_car_in_front'] = self._lap_time[in_front].squeeze()
                data['gap_in_front'] = (self._cumulative_time[index] - self._cumulative_time[in_front]).squeeze()
                # If cars are less than a second distant, the drivers are allowed to use DRS
                if data['gap_in_front'].values[0] < 1000:
                    data['drs'] = 1
                # Battle feature is considered True when the cars are less than 2s afar
                if data['gap_in_front'].values[0] < 2000:
                    data['battle'] = 1
            else:
                data['delta_car_following'] = 0
                data['time_car_in_front'] = 0
                data['gap_in_front'] = 0

            # Compute gaps and features to following car, if any
            if following > - 1:
                data['delta_car_following'] = (self._lap_time[following] - self._lap_time[index]).squeeze()
                data['time_car_following'] = (self._lap_time[following]).squeeze()
                data['gap_following'] = (self._cumulative_time[following] - self._cumulative_time[index]).squeeze()
                # Battle feature is considered True when the cars are less than 2s afar
                if data['gap_following'].values[0] < 2000:
                    data['battle'] = 1
            else:
                data['delta_car_following'] = 0
                data['time_car_in_front'] = 0
                data['gap_following'] = 0

            data['prev_milliseconds_delta'] = (self._lap_deltas[index]).squeeze()
            data['prev_milliseconds'] = (self._old_lap_time[index]).squeeze()
            data['safety'] = safety_laps

            if driver in self._active_drivers and started:  # Driver is controlled by an agent
                active_index = self._active_drivers_mapping[driver]
                # Blank out the original tyre, we want to place the one selected by the action
                data['tyre_1'] = 0
                data['tyre_2'] = 0
                data['tyre_3'] = 0
                data['tyre_4'] = 0
                data['tyre_5'] = 0
                data['tyre_6'] = 0
                data['pit-cost'] = 0
                data['pitstop-milliseconds'] = 0

                if actions[active_index] == 0:  # stay on track, do not pit
                    self.__update_pit_flags(index)

                elif actions[active_index] == 1:  # pit for soft tyre
                    self.__pit_driver(driver, self._soft_tyre, safety_laps)

                elif actions[active_index] == 2:  # pit for medium tyre
                    self.__pit_driver(driver, self._medium_tyre, safety_laps)

                else:  # pit for hard tyre
                    self.__pit_driver(driver, self._hard_tyre, safety_laps)

                data['pitstop-milliseconds'] = self._pit_costs[index]
                data['pit-cost'] = self._pit_costs[index] / 2
                data['tyre_age'] = self._tyre_age[index] / self._race_length
                data['lap'] = (self._t + self.start_lap) / self._race_length
                data[self._current_tyres[index]] = 1
                data['stop'] = self._pit_counts[index]

            elif is_out:  # Crashed driver, non-controlled
                current_tyre = self._current_tyres[index]
                strategy = self._default_strategies[index]

                # Resort to a pre-defined strategy
                if strategy[current_tyre] == int(self._tyre_age[index] * self._race_length):  # Time to pit
                    tyre_list = [self._soft_tyre, self._medium_tyre, self._hard_tyre]
                    tyre_duration = [strategy[self._soft_tyre],
                                     strategy[self._medium_tyre],
                                     strategy[self._hard_tyre]]
                    remaining_laps = self._race_length - self._lap
                    # Select the tyre that can cover most of the remaining laps without overshooting
                    tyre_index = np.argmin(np.abs(remaining_laps - np.asarray(tyre_duration)))
                    selected_tyre = tyre_list[tyre_index[0]]
                    self.__pit_driver(driver, selected_tyre, safety_laps)
                else:  # Stay out
                    self.__update_pit_flags(index)

            else:  # Non-controlled driver, not crashed
                self._pit_states[index] = data['pit']
                self._pit_counts[index] = data['stop']
                self._current_tyres[index] = get_current_tyres(data)
            state = compute_state(data)
            # prediction_model = self._model.get_prediction_model(state)

            # Normalize and remove unnecessary columns
            #data = self._model.normalize_dataset(data)
            #data = data.drop(columns=['unnorm_lap', 'race_length', 'raceId', 'driverId', 'nextLap'])

            #data = fix_data_types(data)

            # if not (state == 'pit' or state == 'safety'):
            #     data = data.drop(columns=['pit', 'safety', 'pitstop-milliseconds', 'pit-cost'])

            #stop = time.time()
            #contribute['features'] += stop - start

            #start = time.time()

            if started:
                if state == "pit":
                    pitting.append(index)
                    x_pit_data.append(data)
                else:
                    no_pit.append(index)
                    x_data.append(data)

                #stop = time.time()
                #contribute['append'] += stop - start

            else:
                self._next_lap_time[index] = row.squeeze()['nextLap']

            # Predict the delta wrt pole lap
            # predicted_lap = prediction_model.predict(data).squeeze()

            # next_lap_time_rf = base_time + predicted_rf

        #start = time.time()
        if started:
            if len(pitting) > 0:
                x_pit_data = pd.concat(x_pit_data)
                self._model.normalize_dataset(x_pit_data)
                x_pit_data = x_pit_data.drop(columns=['unnorm_lap', 'race_length', 'raceId', 'driverId', 'nextLap'])
                pit_predictions = self._model.pit_model.predict(x_pit_data)
                for index, prediction in zip(pitting, pit_predictions):
                    self._next_lap_time[index] = np.random.normal(self._base_time + prediction, 100)

            if len(no_pit) > 0:
                x_data = pd.concat(x_data)
                self._model.normalize_dataset(x_data)
                x_data = x_data.drop(columns=['unnorm_lap', 'race_length', 'raceId', 'driverId', 'nextLap'])
                if safety_laps:
                    prediction_model = self._model.safety_model
                else:
                    x_data = x_data.drop(columns=['pit', 'safety', 'pitstop-milliseconds', 'pit-cost'])
                    prediction_model = self._model.regular_model
                predictions = prediction_model.predict(x_data)
                for index, prediction in zip(no_pit, predictions):
                    self._next_lap_time[index] = np.random.normal(self._base_time + prediction, 100)

        for driver in self._active_drivers:
            active_index = self._active_drivers_mapping[driver]
            index = self._drivers_mapping[driver]
            self._reward[active_index] = -np.clip(self._lap_time[index], 0, self.max_lap_time)

        if self.scale_reward:
            self._reward /= self.max_lap_time
            if self.positive_reward:
                self._reward = 1 + reward

        self._t += 1
        self._terminal = True if self._t >= self.horizon or lap >= len(self._laps) else False
        # self.state = self.get_state()

        # stop = time.time()
        # contribute['predict'] = stop - start
        # for k, v in contribute.items():
        #     print(k, v)

        return self.get_state(), self._reward, self._terminal, {}

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
                compound_strategy = np.random.normal(tyre_average_stints[compound][0], tyre_average_stints[compound][0])
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
                self._tyre_age[index] = data['tyre_age']
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
                self._tyre_age[index] = data['tyre_age']
                self._current_tyres[index] = get_current_tyres(data)

        for driver in self._active_drivers:
            index = self._drivers_mapping[driver]
            agent_index = self._active_drivers_mapping[driver]
            self._agents_last_pit[agent_index] = int(self._tyre_age[index] * self._race_length)

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
