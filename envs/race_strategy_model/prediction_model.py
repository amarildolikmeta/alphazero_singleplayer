from copy import deepcopy

import pandas as pd
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBRegressor
import numpy as np

CIRCUIT_LIST = ["circuitId_1", "circuitId_2", "circuitId_3", "circuitId_4", "circuitId_6",
                "circuitId_7", "circuitId_9", "circuitId_10", "circuitId_11", "circuitId_13", "circuitId_14",
                "circuitId_15", "circuitId_17", "circuitId_18", "circuitId_22", "circuitId_24", "circuitId_32",
                "circuitId_34", "circuitId_69", "circuitId_70", "circuitId_71", "circuitId_73", "tyre_1", "tyre_2"]

def get_current_circuit(df: pd.DataFrame):
    for circuit in CIRCUIT_LIST:
        if df[circuit].all() == 1:
            return circuit

    raise ValueError('Something wrong with the race dataframe, multiple circuits in the same race')


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
        if not (race_laps['tyre_7'].all() == 0 and race_laps['tyre_8'].all() == 0 and race_laps['rainy'].all() == 0):
            data = data.loc[data['raceId'] != race]

    # Drop all wet-related information from the dataframe
    data = data.drop(columns=['tyre_7', 'tyre_8', 'rainy'])

    return data


def discard_suspended_races(data):
    """ Remove races containing laps slower than double the pole lap, they have probably been interrupted """

    # Divide by the pole time
    data['nextLap'] = data['nextLap'] / data['pole']

    # Find lap times abnormally high
    anomalies = data[data['nextLap'] > 2]
    races_to_discard = anomalies['raceId'].unique()

    for race in races_to_discard:
        data = data[data['raceId'] != race]

    # Undo the pole time division
    data['nextLap'] = data['nextLap'] * data['pole']
    return data


class RaceStrategyModel(object):
    def __init__(self, year: int, verbose=False):
        self.regular_model = None
        self.pit_model = None
        self.safety_model = None
        self.test_race = None
        self.scaler = None
        self.test_race_pit_model = None
        # self.start_lap = start_lap

        if year == 2014:
            year = "year_1"
        elif year == 2015:
            year = "year_2"
        elif year == 2016:
            year = "year_3"
        elif year == 2017:
            year = "year_4"
        elif year == 2018:
            year = "year_5"
        elif year == 2019:
            year = "year_6"
        else:
            raise ValueError("No race available for year " + str(year))

        self.year = year
        self.verbose = verbose

    def split_train_test(self, df: pd.DataFrame, split_fraction: float):
        """ Split the dataset randomly but keeping whole races together """
        test_data = pd.DataFrame(columns=df.columns)

        races = df[df[self.year] == 1]['raceId'].unique()

        if split_fraction != 0:
            split_size = int(round(split_fraction * len(races)))
        else:
            # Leave only one race out from the training
            split_size = 1

        test_races = np.random.choice(races, size=split_size)
        for race in test_races:
            race_laps = df.loc[df['raceId'] == race]
            test_data = test_data.append(race_laps)
            df = df[df.raceId != race]

        return df, test_data

    def normalize_dataset(self, df):
        """ Normalize integer-valued columns of the dataset """
        data = df.copy()
        # print(df.columns)
        # Remove columns not to be normalized
        zero_one = ['battle', 'drs', "circuitId_1", "circuitId_2", "circuitId_3", "circuitId_4", "circuitId_6",
                    "circuitId_7", "circuitId_9", "circuitId_10", "circuitId_11", "circuitId_13", "circuitId_14",
                    "circuitId_15", "circuitId_17", "circuitId_18", "circuitId_22", "circuitId_24", "circuitId_32",
                    "circuitId_34", "circuitId_69", "circuitId_70", "circuitId_71", "circuitId_73", "tyre_1", "tyre_2",
                    "tyre_3", "tyre_4", "tyre_5", "tyre_6",
                    "year_1", "year_2", "year_3", "year_4", "year_5", "year_6", "nextLap", 'pit', 'safety', "unnorm_lap"]
                    #'milliseconds',
                    #'cumulative', 'unnorm_lap']

        temp_df = data[zero_one].copy()
        data.drop(zero_one, axis=1, inplace=True)

        # if self.columns is not None and len(data.columns) != len(self.columns):
        #     print(set(data.columns).difference(set(self.columns)))
        #     exit(-1)

        if not self.scaler:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(data)
            scaled = data
        else:
            scaled = self.scaler.transform(data)
        data.loc[:, :] = scaled
        data = data.join(temp_df)

        del temp_df
        return data

    def __process_dataset(self, dataset):
        """ Pre-process the dataset to obtain training data and its labels"""

        # Discard wet and suspended races
        old_races = len(dataset['raceId'].unique())
        dataset = discard_wet(dataset)
        dataset = discard_suspended_races(dataset)
        new_races = len(dataset['raceId'].unique())
        if self.verbose:
            print("{} wet and suspended races were discarded".format(old_races - new_races))

        # Eliminate the last lap from the training data, as it has 0 target
        dataset = dataset[dataset['nextLap'] > 0]

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
        dataset = self.normalize_dataset(dataset)

        _, self.test_race = self.split_train_test(dataset, split_fraction=0)
        self.__compute_pitstop_model(dataset)

        train_data = self.normalize_dataset(dataset)

        # train_data = train_data[train_data['unnorm_lap'] > self.start_lap]  # Take laps after a threshold

        # Remove columns used only to identify the laps in testing
        train_data = train_data.drop(columns=['unnorm_lap', "raceId", "driverId", "race_length"])

        # Split the dataset into three separate datasets, one per each model to be trained
        train_pit = deepcopy(train_data.loc[train_data['pit'] != 0])
        train_safety = deepcopy(train_data.loc[(train_data['safety'] != 0) & (train_data['pit'] == 0)])
        train_regular = deepcopy(train_data.loc[(train_data['pit'] == 0) & (train_data['safety'] == 0)])

        # Remove features related to pit and safety in the "regular" laps model
        train_regular = train_regular.drop(columns=['safety', 'pit', 'pit-cost', 'pitstop-milliseconds'])

        # Extract the target labels
        labels_pit = train_pit.pop('nextLap')
        labels_safety = train_safety.pop('nextLap')
        labels_regular = train_regular.pop('nextLap')

        train_data = {'regular': train_regular, 'safety': train_safety, 'pit': train_pit}
        labels = {'regular': labels_regular, 'safety': labels_safety, 'pit': labels_pit}

        return train_data, labels

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


    def train(self):
        """ Train the regression models """
        if self.verbose:
            print('Training models...')
        self.scaler = None
        self.regular_model = XGBRegressor()
        self.pit_model = XGBRegressor()
        self.safety_model = XGBRegressor()

        dataset = load_dataset()
        datasets, labels = self.__process_dataset(dataset)

        self.regular_model.fit(datasets['regular'], labels['regular'])
        self.pit_model.fit(datasets['pit'], labels['pit'])
        self.safety_model.fit(datasets['safety'], labels['safety'])

        if self.verbose:
            print('Done!\n')

    def resplit(self):
        # TODO fix the invalidation of scaler to avoid the normalization of test races
        self.scaler = None
        dataset = load_dataset()
        self.__process_dataset(dataset)

    def load(self):
        """ Restore prediction models from previously pickled files to avoid retraining """

        if self.verbose:
            print("Loading prediction models from pickled files...")
        if not os.path.isfile("./envs/race_strategy_model/pickled_models/regular.pickle"):
            print("ERROR: regular.pickle is missing")
            exit(-1)
        else:
            with open('./envs/race_strategy_model/pickled_models/regular.pickle', 'rb') as regular_file:
                self.regular_model = pickle.load(regular_file)
                regular_file.close()

        if not os.path.isfile("./envs/race_strategy_model/pickled_models/safety.pickle"):
            print("ERROR: safety.pickle is missing")
            exit(-1)
        else:
            with open('./envs/race_strategy_model/pickled_models/safety.pickle', 'rb') as safety_file:
                self.safety_model = pickle.load(safety_file)
                safety_file.close()

        if not os.path.isfile("./envs/race_strategy_model/pickled_models/pit.pickle"):
            print("ERROR: pit.pickle is missing")
            exit(-1)
        else:
            with open('./envs/race_strategy_model/pickled_models/pit.pickle', 'rb') as pit_file:
                self.pit_model = pickle.load(pit_file)
                pit_file.close()

        if not os.path.isfile("./envs/race_strategy_model/pickled_models/scaler.pickle"):
            print("ERROR: scaler.pickle is missing")
            exit(-1)
        else:
            with open('./envs/race_strategy_model/pickled_models/scaler.pickle', 'rb') as scaler_file:
                self.scaler = pickle.load(scaler_file)
                scaler_file.close()

        # if not os.path.isfile("pickled_models/test_race.pickle"):
        #     print("ERROR: test_race.pickle is missing")
        #     exit(-1)
        # else:
        #     with open('pickled_models/test_race.pickle', 'rb') as pit_file:
        #         self.pit_model = pickle.load(pit_file)
        #         pit_file.close()

        if self.verbose:
            print("Done!\n")

    def save(self):
        """ Pickle the model objects to avoid retraining """

        for model, name in zip([self.regular_model, self.safety_model, self.pit_model, self.scaler],
                               ['regular', 'safety', 'pit', 'scaler']):
            with open('./envs/race_strategy_model/pickled_models/{}.pickle'.format(name), 'wb') as savefile:
                pickle.dump(model, savefile)
                savefile.close()
        #self.test_race.to_csv(".envs/race_strategy_model/dataset/test_race.csv")

    def predict(self, state, lap_type):
        if lap_type == 'regular':
            state.drop(columns=['safety', 'pit', 'pit-cost', 'pitstop-milliseconds'])
            return self.regular_model.predict(state)
        elif lap_type == 'pit':
            return self.regular_model.predict(state)
        else:
            return self.safety_model.predict(state)

    def get_prediction_model(self, state: str):
        if state == 'regular':
            return self.regular_model
        if state == 'safety':
            return self.safety_model
        if state == 'pit':
            return self.pit_model
        else:
            raise ValueError("The specified state is not valid, allowed model states are 'regular', 'safety' and 'pit'")



if __name__ == '__main__':
    model = RaceStrategyModel(2019)
    #model.load()
    #model.resplit()
    model.train()
    model.save()

    print(model.test_race['driverId'].unique())
