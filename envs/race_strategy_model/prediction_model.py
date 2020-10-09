import copy
from collections import defaultdict
from copy import deepcopy
import time
import pandas as pd
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBRegressor
import numpy as np
import tensorflow as tf

CIRCUIT_LIST = ["circuitId_1", "circuitId_2", "circuitId_3", "circuitId_4", "circuitId_6",
                "circuitId_7", "circuitId_9", "circuitId_10", "circuitId_11", "circuitId_13", "circuitId_14",
                "circuitId_15", "circuitId_17", "circuitId_18", "circuitId_22", "circuitId_24", "circuitId_32",
                "circuitId_34", "circuitId_69", "circuitId_70", "circuitId_71", "circuitId_73", "tyre_1", "tyre_2"]

def get_current_circuit(df: pd.DataFrame):
    for circuit in CIRCUIT_LIST:
        if (df[circuit] == 1).all():
            return circuit

    raise ValueError('Something wrong with the race dataframe, multiple circuits in the same race')

def fix_data_types(to_fix):
    for col in to_fix.columns:
        if to_fix[col].dtype == 'object':
            to_fix[col] = to_fix[col].astype(str).astype(int)

    return to_fix

def load_dataset():
    """ Load the dataset and build the pandas dataframe """

    # TODO load from absolute path
    data_complete = pd.read_csv('./envs/race_strategy_model/dataset/finalDataset.csv', delimiter=',')

    data_complete = pd.concat([data_complete, pd.get_dummies(data_complete['tyre'], prefix='tyre')], axis=1).drop(
        ['tyre'], axis=1)
    data_complete = pd.concat([data_complete, pd.get_dummies(data_complete['circuitId'], prefix='circuitId')],
                              axis=1).drop(['circuitId'], axis=1)
    data_complete = pd.concat([data_complete, pd.get_dummies(data_complete['year'], prefix='year')], axis=1).drop(
        ['year'], axis=1)

    return data_complete


def discard_wet(data: pd.DataFrame):
    """ Discard the wet races, as we don't have enough data to predict correctly
    the wetness and performance  of the track"""

    races = data['raceId'].unique()

    for race in races:
        race_laps = data[data['raceId'] == race]

        # Select only races having for each lap all the flags related to wet conditions set to 0
        if not ((race_laps['tyre_7'] == 0).all() and (race_laps['tyre_8'] == 0).all() and (race_laps['rainy'] == 0).all()):
            data = data.loc[data['raceId'] != race]

    # Drop all wet-related information from the dataframe
    data = data.drop(columns=['tyre_7', 'tyre_8', 'rainy'])

    return data


def discard_suspended_races(data):
    """ Remove races containing laps slower than double the pole lap, they have probably been interrupted """

    # Divide by the pole time
    data['nextLap'] = data['nextLap'] / data['pole']

    # Find lap times abnormally high
    anomalies = data[data['nextLap'] > 3]
    races_to_discard = anomalies['raceId'].unique()

    for race in races_to_discard:
        data = data[data['raceId'] != race]

    # Undo the pole time division
    data['nextLap'] = data['nextLap'] * data['pole']
    return data

def discard_wet_quali(df):
    """Discard races whose qualifying sessions were held on wet conditions"""
    # print("Races before cut:", len(df['raceId'].unique()))
    df = df.loc[df['wet_quali'] == 0]
    # print("Races after cut:", len(df['raceId'].unique()))
    return df


class RaceStrategyModel(object):
    def __init__(self, year: int, start_lap:int, verbose=False, n_cores=1, scalers=None):
        # self.regular_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/regular.h5')
        # self.pit_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/pit.h5')
        # self.safety_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/safety.h5')
        self.regular_model = XGBRegressor(n_jobs=n_cores)
        self.pit_model = XGBRegressor(n_jobs=n_cores)
        self.safety_model = XGBRegressor(n_jobs=n_cores)
        self.regular_model.load_model('./envs/race_strategy_model/pickled_models/xgb_regular.model')
        self.pit_model.load_model('./envs/race_strategy_model/pickled_models/xgb_pit.model')
        self.safety_model.load_model('./envs/race_strategy_model/pickled_models/xgb_safety.model')

        self.test_race = None
        self.scaler = None
        self.test_race_pit_model = None
        self.dummy_columns = None
        self.n_cores = n_cores
        self.start_lap = start_lap
        self.ready_dataset = None
        self.laps_database = defaultdict(lambda: None)
        self.race_id = -1
        # self.start_lap = start_lap

        if year == 2014:
            self.year = "year_1"
        elif year == 2015:
            self.year = "year_2"
        elif year == 2016:
            self.year = "year_3"
        elif year == 2017:
            self.year = "year_4"
        elif year == 2018:
            self.year = "year_5"
        elif year == 2019:
            self.year = "year_6"
        else:
            raise ValueError("No race available for year " + str(year))

        self.year_numeric = year
        self.verbose = verbose

        if not scalers:
            dataset = load_dataset()
            self.target_scalers = self.__process_dataset(dataset)
        else:
            self.target_scalers = scalers

    # def __deepcopy__(self, memodict={}):
    #     new = RaceStrategyModel(self.year_numeric, self.start_lap, verbose=self.verbose,
    #                             n_cores=self.n_cores, scalers=self.target_scalers)
    #
    #     new.regular_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/regular.h5')
    #     new.pit_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/pit.h5')
    #     new.safety_model = tf.keras.models.load_model('./envs/race_strategy_model/pickled_models/safety.h5')
    #     new.test_race = self.test_race
    #     new.scaler = self.scaler
    #     new.test_race_pit_model = self.test_race_pit_model
    #     new.dummy_columns = self.dummy_columns
    #     new.ready_dataset = self.ready_dataset
    #     new.laps_database = self.laps_database
    #     new.race_id = self.race_id
    #     return new

    def split_train_test(self, df: pd.DataFrame, split_fraction: float):
        """ Split the dataset randomly but keeping whole races together """
        data = df.copy()
        test_data = pd.DataFrame(columns=df.columns)

        races = data[data[self.year] == 1]['raceId'].unique()

        if split_fraction != 0:
            split_size = int(round(split_fraction * len(races)))
        else:
            # Leave only one race out from the training
            split_size = 1

        test_races = np.random.choice(races, size=split_size)
        for race in test_races:
            race_laps = data.loc[data['raceId'] == race]
            test_data = test_data.append(race_laps)
            data = data[data.raceId != race]

        return data, test_data

    def normalize_dataset(self, df, compute_masks=True):
        """ Normalize integer-valued columns of the dataset """
        data = df.copy()
        regular_mask, pit_mask, safety_mask = None, None, None
        # Remove columns not to be normalized
        zero_one = ['nextLap']

        if compute_masks:
            safety_mask = (data.safety > 0) & (data.pit == 0)
            pit_mask = data.pit != 0
            regular_mask = (data.pit == 0) & (data.safety == 0)

        temp_df = data[zero_one].copy()
        data.drop(zero_one, axis=1, inplace=True)

        if not self.scaler:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler = self.scaler.fit(data)
            scaled = data
        else:
            scaled = self.scaler.transform(data)
        data.loc[:, :] = scaled
        data = data.join(temp_df)
        del temp_df
        return data, regular_mask, pit_mask, safety_mask

    def __process_dataset(self, dataset):
        """ Pre-process the dataset to obtain training data and its labels"""

        # Eliminate the last lap from the training data, as it has 0 target
        dataset = dataset[dataset['nextLap'] > 0]
        #dataset = dataset[dataset['unnorm_lap'] > self.start_lap]

        # Discard wet and suspended races
        old_races = len(dataset['raceId'].unique())
        # dataset = discard_suspended_races(dataset)
        dataset = discard_wet_quali(dataset)
        dataset = discard_wet(dataset)
        new_races = len(dataset['raceId'].unique())
        # print(dataset.lap.count())
        if self.verbose:
            print("{} wet and suspended races were discarded".format(old_races - new_races))

        # Express the next lap target as a delta to the pole lap
        dataset['nextLap'] = (dataset['nextLap'] - dataset['pole'])

        # Duplicate columns to use them after normalization
        dataset['base'] = dataset['pole'].astype(int)
        dataset['true'] = dataset['milliseconds'].astype(int)
        dataset['true_cumulative'] = dataset['cumulative'].astype(int)

        # Normalize the dataset, but normalize the lap time and cumulative time individually, in order to be able to
        # normalize them at runtime

        # Remove the duplicated unnormalized columns from the train data
        dataset = dataset.drop(columns=['base', 'true', 'true_cumulative'])
        dataset, _, _, _ = self.normalize_dataset(dataset, compute_masks=False)
        self.ready_dataset = copy.deepcopy(dataset)

        self.dummy_columns = dataset.columns
        norm_data, reg_mask, pit_mask, safety_mask = self.normalize_dataset(dataset)


        # Remove columns used only to identify the laps in testing
        norm_data = norm_data.drop(columns=['unnorm_lap', "raceId", "driverId"])

        # Split the dataset into three separate datasets, one per each model to be trained
        train_pit = deepcopy(norm_data.loc[pit_mask])
        train_safety = deepcopy(norm_data.loc[safety_mask])
        train_regular = deepcopy(norm_data.loc[reg_mask])

        # Eliminate outlier data
        train_pit = train_pit.loc[train_pit['nextLap'] < train_pit['nextLap'].quantile(0.94)]
        train_safety = train_safety.loc[train_safety['nextLap'] <  train_safety['nextLap'].quantile(0.97)]
        train_regular = train_regular.loc[train_regular['nextLap'] <  train_regular['nextLap'].quantile(0.97)]

        # Remove features related to pit and safety in the "regular" laps model
        train_regular = train_regular.drop(columns=['safety', 'pit', 'pit-cost', 'pitstop-milliseconds'])

        # Extract the target labels
        labels_pit = train_pit.pop('nextLap')
        labels_safety = train_safety.pop('nextLap')
        labels_regular = train_regular.pop('nextLap')

        scaler_regular = MinMaxScaler(feature_range=(-1,1))
        scaler_pit = MinMaxScaler(feature_range=(-1,1))
        scaler_safety = MinMaxScaler(feature_range=(-1,1))

        scaler_pit = scaler_pit.fit(labels_pit.to_frame())
        scaler_safety = scaler_safety.fit(labels_safety.to_frame())
        scaler_regular = scaler_regular.fit(labels_regular.to_frame())

        # print(labels_regular.describe())
        # exit()

        scalers = {'regular': scaler_regular, 'safety': scaler_safety, 'pit': scaler_pit}

        return scalers

    def __compute_pitstop_model(self, full_dataset: pd.DataFrame):
        """Compute a normal distribution's parameters for each driver's pit-stop times"""
        circuit = get_current_circuit(self.test_race)

        pits = []
        pits_safety = []

        stop_laps = full_dataset[(full_dataset['pitstop-milliseconds'] > 0) & (full_dataset[circuit] == 1)].sort_values('lap')

        pit_times = stop_laps[stop_laps['safety'] == 0]['pitstop-milliseconds'].values
        pit_safety_times = stop_laps[stop_laps['safety'] > 0]['pitstop-milliseconds'].values
        pits.extend(pit_times.tolist())
        pits_safety.extend(pit_safety_times.tolist())

        safety_mean = np.mean(pit_safety_times) if len(pit_safety_times) > 0 else 0
        safety_std = np.std(pit_safety_times) if len(pit_safety_times) > 0 else 0

        mean = np.mean(pit_times) if len(pit_times) > 0 else 0
        std = np.std(pit_times) if len(pit_times) > 0 else 0

        self.test_race_pit_model = {'regular': (mean, std), 'safety': (safety_mean, safety_std)}

    def resplit(self):
        # TODO fix the invalidation of scaler to avoid the normalization of test races
        _, self.test_race = self.split_train_test(self.ready_dataset, 0)
        self.test_race = fix_data_types(self.test_race)
        self.race_id = self.test_race["raceId"].values[0]
        self.__compute_pitstop_model(self.ready_dataset)

        # Insert single rows in a dictionary for faster access
        for i in range(self.test_race["lap"].count()):
            row = self.test_race.iloc[[i]]
            self.laps_database[(row["driverId"].values[0], row["lap"].values[0])] = row


if __name__ == '__main__':
    model = RaceStrategyModel(2019)
    #model.load()
    #model.resplit()
    model.train()
    model.save()

    print(model.test_race['driverId'].unique())
