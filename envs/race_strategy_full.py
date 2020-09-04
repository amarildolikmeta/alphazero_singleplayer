import csv

import gym
from copy import copy
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import register
from envs.race_strategy_model.train_model import RaceStrategyModel
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
        tyre_stints[compound] = (np.mean(tyre_stints[compound]), np.std(tyre_stints[compound]))

    return tyre_stints


def get_pit_stop_models(race):
    pit_milli_model = {}
    for driver in race['driverId'].unique():
        stop_laps = race[(race['driverId'] == driver) & (race['pitstop-milliseconds'] > 0)].sort_values('lap')
        pit_milli_model[driver] = (
            np.mean(stop_laps['pitstop-milliseconds'].values), np.std(stop_laps['pitstop-milliseconds'].values))

    return pit_milli_model


def compute_state(row):
    if row['pit'].all() != 0:
        return 'pit'
    elif row['safety'].all() != 0:
        return 'safety'
    else:
        return 'regular'


def fix_data_types(to_fix):
    for col in to_fix.columns:
        if to_fix[col].dtype == 'object':
            to_fix[col] = to_fix[col].astype(str).astype(int)

    return to_fix


class RaceModel(gym.Env):

    def __init__(self, gamma=0.95, horizon=20, scale_reward=False, positive_reward=True, start_lap=8):

        self.horizon = horizon
        self.gamma = gamma
        self.obs_dim = 7
        self.action_space = spaces.Discrete(n=4)
        self.observation_space = spaces.Box(low=0., high=self.horizon,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.scale_reward = scale_reward
        self.positive_reward = positive_reward
        self.viewer = None

        with open('./race_strategy_model/active_drivers.csv', newline='') as f:
            reader = csv.reader(f)
            line = reader.__next__()
            self._active_drivers = np.asarray(line[1:], dtype=int)
            self._year = int(line[0])
            f.close()

        self._active_drivers = set(self._active_drivers)


        self.start_lap = start_lap
        self._lap = 0

        self._t = -start_lap

        self._model = None

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

        self._pit_model = None

        self._drivers_mapping = {}
        self._active_drivers_mapping = {}

        self.seed()
        self.reset()

        if self.scale_reward:
            print("Reward is being normalized")

    def get_state(self):
        state = [self._lap,
                 self._current_tyres,
                 self._cumulative_time,
                 self._lap_time,
                 self._pit_states,
                 self._pit_counts,
                 self._tyre_age]
        return np.asarray(state)

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
        sig = {'state': np.copy(self.get_state()),
               'next_lap_time': np.copy(self._next_lap_time),
               'last_row': copy(self._last_available_row),
               }
        return sig

    def set_signature(self, sig):
        self.__set_state(sig["state"])
        self._next_lap_time = sig['next_lap_time']
        self._last_available_row = sig['last_row']

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __pit_driver(self, driver, tyre):
        index = self._drivers_mapping[driver]
        self._pit_states[index] = -1
        self._current_tyres[index] = tyre
        self._pit_counts[index] += 1
        self._pit_costs[index] = np.random.normal(self._pit_model[driver][0],
                                                  self._pit_model[driver][1])

    def __update_pit_flags(self, index):
        if self._pit_states[index] == -1:
            self._pit_states[index] = 1
            self._tyre_age[index] = 0

        elif self._pit_states[index] == 1:
            self._pit_states[index] = 0
            self._pit_costs[index] = 0

    def step(self, actions: np.ndarray):

        assert len(actions) == len(self._active_drivers), \
            "Fewer actions were provided than the number of active drivers"

        reward = np.zeros(len(actions))
        if self._t >= self.horizon:
            return self.get_state(), 0, True, {}

        lap = self.start_lap + self._t + 1
        self._lap = lap
        lap_norm = self._laps[lap - 1]

        # Update current lap times and compute delta wrt previous lap
        self._old_lap_time = self._lap_time.copy()
        self._lap_time = self._next_lap_time.copy()
        self._lap_deltas = self._lap_time - self._old_lap_time
        self._cumulative_time = self._cumulative_time + self._lap_time

        # Compute current positions for XGBoost predicted cumulative times
        temp = np.argsort(self._cumulative_time)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(self._cumulative_time)) + 1

        # Simulate each driver
        for driver in self._drivers_mapping:
            index = self._drivers_mapping[driver]

            # Take the lap for the current driver
            race_laps = self._model.test_race
            row = race_laps[(race_laps['driverId'] == driver) & (race_laps['lap'] == lap_norm)].sort_values('lap')

            # The driver might have retired if no row is available in the dataset
            is_out = False
            if row['lap'].count() == 0:
                is_out = True
                row = self._last_available_row[index]
            else:
                self._last_available_row[index] = row.copy()

            if lap < self.start_lap and not is_out:
                self._lap_time[index] = row.squeeze()['milliseconds']

            # Find the index of the drivers following and preceding the current car
            position = ranks[index]

            in_front = None
            following = None

            if position > 1:  # The leader has no car in front, consider only other cars
                in_front = np.argwhere(ranks == position - 1)

            if position < self._drivers_number and not (
                    self._cumulative_time[
                        np.argwhere(ranks == position + 1)] == np.inf):  # The last car has no follower
                following = np.argwhere(ranks == position + 1)

            # Fill the template dataframe with the features extracted from predicted time
            data = row.copy()
            data['milliseconds'] = self._lap_time[index].squeeze()
            data['cumulative'] = self._cumulative_time[index].squeeze()
            data['position'] = position
            data['drs'] = 0
            data['battle'] = 1

            # Compute gaps and features to car in front, if any
            if in_front is not None:
                data['delta_car_in_front'] = (self._lap_time[index] - self._lap_time[in_front]).squeeze()
                data['time_car_in_front'] = self._lap_time[in_front].squeeze()
                data['gap_in_front'] = (self._cumulative_time[index] - self._cumulative_time[in_front]).squeeze()
                # If cars are less than a second of distance, the drivers are allowed to use DRS
                if data['gap_in_front'].all() < 1000:
                    data['drs'] = 1
                if data['gap_in_front'].all() < 2000:
                    data['battle'] = 1
            else:
                data['delta_car_following'] = 0
                data['time_car_in_front'] = 0
                data['gap_in_front'] = 0

            # Compute gaps and features to following car, if any
            if following is not None:
                data['delta_car_following'] = (self._lap_time[following] - self._lap_time[index]).squeeze()
                data['time_car_following'] = (self._lap_time[following]).squeeze()
                data['gap_following'] = (self._cumulative_time[following] - self._cumulative_time[index]).squeeze()
                if data['gap_following'].all() < 2000:
                    data['battle'] = 1
            else:
                data['delta_car_following'] = 0
                data['time_car_in_front'] = 0
                data['gap_following'] = 0

            data['prev_milliseconds_delta'] = (self._lap_deltas[index]).squeeze()
            data['prev_milliseconds'] = (self._old_lap_time[index]).squeeze()

            if driver in self._active_drivers and lap > self.start_lap:  # Driver is controlled by an agent
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
                    self.__pit_driver(driver, self._soft_tyre)

                elif actions[active_index] == 2:  # pit for medium tyre
                    self.__pit_driver(driver, self._medium_tyre)

                else:  # pit for hard tyre
                    self.__pit_driver(driver, self._hard_tyre)

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
                if strategy[current_tyre] == self._tyre_age[index]:  # Time to pit
                    tyre_list = [self._soft_tyre, self._medium_tyre, self._hard_tyre]
                    tyre_duration = [strategy[self._soft_tyre],
                                     strategy[self._medium_tyre],
                                     strategy[self._hard_tyre]]
                    remaining_laps = self._race_length - self._lap
                    # Select the tyre that can cover most of the remaining laps without overshooting
                    tyre_index = np.argmin(np.abs(remaining_laps - np.array(tyre_duration)))
                    selected_tyre = tyre_list[tyre_index[0]]
                    self.__pit_driver(driver, selected_tyre)
                else:  # Stay out
                    self.__update_pit_flags(index)

            else:  # Non-controlled driver, not crashed
                self._pit_states[index] = data['pit']
                self._pit_counts[index] = data['stop']
                self._current_tyres[index] = get_current_tyres(data)

            state = compute_state(data)

            prediction_model = self._model.get_prediction_model(state)

            # Normalize and remove unnecessary columns
            data = self._model.normalize_dataset(data)
            data = data.drop(columns=['unnorm_lap', 'race_length', 'raceId', 'driverId', 'nextLap'])

            data = fix_data_types(data)

            if not (state == 'pit' or state == 'safety'):
                data = data.drop(columns=['pit', 'safety', 'pitstop-milliseconds', 'pit-cost'])

            # Predict the delta wrt pole lap
            predicted_lap = prediction_model.predict(data).squeeze()

            # predicted_rf = model_rf.predict(data).squeeze()

            if lap > self.start_lap:
                self._next_lap_time[index] = np.random.normal(self._base_time + predicted_lap, 100)
            else:
                self._next_lap_time[index] = row.squeeze()['nextLap']
            # next_lap_time_rf = base_time + predicted_rf

            if driver in self._active_drivers:
                active_index = self._active_drivers_mapping[driver]
                reward[active_index] = -np.clip(self._lap_time[index], 0, self.max_lap_time)

        if self.scale_reward:
            reward /= self.max_lap_time

        self._t += 1
        terminal = True if self._t >= self.horizon else False
        # self.state = self.get_state()
        if self.positive_reward:
            reward = 1 + reward

        return self.get_state(), reward, terminal, {}

    def reset(self):
        self._t = -self.start_lap
        self._model = RaceStrategyModel(self._year, self.start_lap)
        self._model.train()

        # Take the base time, the predictions will be deltas from this time
        self._base_time = self._model.test_race['pole'].values[0]
        self.max_lap_time = self._base_time * 5
        self._drivers = self._model.test_race['driverId'].unique()
        self._drivers_number = len(self._drivers)
        self._race_length = int(self._model.test_race['race_length'].values[0])
        self._default_strategies = None
        self._laps = self._model.test_race.sort_values('lap')['lap'].unique()

        self._drivers_number = self._model.test_race['driverId'].count()

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

        self._pit_model = get_pit_stop_models(self._model.test_race)

        assert set(self._active_drivers).issubset(set(self._drivers)), "Not all active drivers were in the current race"
        self._drivers_mapping = {}
        self._active_drivers_mapping = {}

        for driver, index in zip(self._drivers, range(self._drivers_number)):
            self._drivers_mapping[driver] = index

        for driver, index in zip(self._active_drivers, range(self._drivers_number)):
            self._active_drivers_mapping[driver] = index

        self._soft_tyre, self._medium_tyre, self._hard_tyre = find_available_rubber(self._model.test_race)
        tyre_average_stints = get_default_strategies(self._model.test_race)
        self._default_strategies = [None] * self._drivers_number

        for driver in self._drivers_mapping:
            index = self._drivers_mapping[driver]
            strategy = {}
            for compound in [self._soft_tyre, self._medium_tyre, self._hard_tyre]:
                compound_strategy = np.random.normal(tyre_average_stints[compound][0], tyre_average_stints[compound][0])
                if np.isnan(compound_strategy):
                    strategy[compound] = np.inf
                else:
                    strategy[compound] = int(round(compound_strategy))
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

        # Overwrite the predicted data for those that are currently in the race

        for i in range(self.start_lap):
            self.step(np.asarray([0] * len(self._active_drivers)))

        return self.get_state()


register(
    id='RaceStrategy-v1',
    entry_point='envs.race_strategy_full:RaceModel'
)

if __name__ == '__main__':
    mdp = RaceModel()
    _ = mdp.reset()
    ret = 0
    while True:
        # print(s)
        a = np.random.choice([0, 1, 2, 3], 9, replace=True)
        s, r, done, _ = mdp.step(a)
        print("Reward:" + str(r) + " Lap Time: " + str(r * mdp.max_lap_time))
        mdp.set_signature(mdp.get_signature())
        ret += r
        if done:
            print("Return:", ret)
            #print("Race Time:", mdp.time)
            break
