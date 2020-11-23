import errno
import json
import os
import time
from datetime import datetime
from statistics import mean

from matplotlib import pyplot as plt

import numpy as np
import neptune

VERBOSITY_LEVELS = [0,1]

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def __init__(self, show=False, verbosity_level=1, enable_neptune=True, experiment_name=None,
                 workspace_name="diggs/race-strategy"):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.verbosity_level = verbosity_level
        self.save_dir, self.numpy_dumps_dir, self.pickled_dir = None, None, None
        self.is_remote = not show
        self.training_V_loss = []
        self.training_pi_loss = []
        self.enable_neptune=enable_neptune
        self.experiment_name = experiment_name
        self.experiment_params = None
        self.experiments = []
        self.workspace_name = workspace_name

    def set_timestamp(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def set_experiment_name(self, name:str) -> None:
        self.experiment_name = name

    def set_experiment_params(self, params:dict) -> None:
        self.experiment_params = params

    def create_experiment(self) -> int:
        i = len(self.experiments)
        try:
            if self.enable_neptune:
                neptune.init(project_qualified_name=self.workspace_name)
                if self.experiment_name is None:
                    exp = neptune.create_experiment(name="Exp_" + str(i),
                                                        params=self.experiment_params,
                                                        tags=[self.timestamp])
                else:
                    exp = neptune.create_experiment(name="Exp_" + str(i),
                                                    params=self.experiment_params,
                                                    tags=[self.experiment_name, self.timestamp])
                self.experiments.append(exp.id)
        except Exception as e:
            print(e)
        return i

    def set_verbosity_level(self, level: int) -> None:
        if level not in VERBOSITY_LEVELS:
            print("[WARNING]The specified verbosity level is not available, admissible values are {}".format(VERBOSITY_LEVELS))
            print("Defaulting to level", self.verbosity_level)
        else:
            self.verbosity_level = level

    def create_directories(self, game: str, out_dir:str) -> None:
        if out_dir is not None:
            mydir = os.path.join(out_dir, self.timestamp)
        else:
            mydir = os.path.join(os.getcwd(), "logs", game, self.timestamp)
        try:
            os.makedirs(mydir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Something wrong with the creation of the log folder")
                raise OSError # This was not a "directory exist" error..

        try:
            os.makedirs(os.path.join(mydir, "plots"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Something wrong with the creation of the plots folder")
                raise  # This was not a "directory exist" error..

        try:
            os.makedirs(os.path.join(mydir, "numpy_dumps"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Something wrong with the creation of the numpy_dumps folder")
                raise  # This was not a "directory exist" error..

        try:
            os.makedirs(os.path.join(mydir, "pickled"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Something wrong with the creation of the pickled folder")
                raise  # This was not a "directory exist" error..
        self.save_dir, self.numpy_dumps_dir, self.pickled_dir = \
            mydir, os.path.join(mydir, "numpy_dumps"), os.path.join(mydir, "pickled")

        print("Logs directory:", self.save_dir)

    def save_parameters(self, params: dict):
        """Save parameters in JSON format over a txt file"""
        assert params is not None, "[ERROR] None parameter dictionary is not valid to be saved"
        assert self.save_dir is not None, \
            "[ERROR] create_directories must be called after initializing the logger and before starting to log"

        with open(os.path.join(self.save_dir, "parameters.txt"), 'w') as d:
            d.write(json.dumps(params))

        self.experiment_params = params

    def log_episode(self, res) -> None:
        assert hasattr(res, "ep_length")
        assert hasattr(res, "return_per_timestep")
        assert hasattr(res, "action_counts")
        assert hasattr(res, "return_per_timestep")
        assert hasattr(res, "execution_time")

        if self.enable_neptune:
            workspace = neptune.init(project_qualified_name=self.workspace_name)
            exp = workspace.get_experiments(id=res.episode_id)[0]
            timestamp = time.time()
            exp.log_metric('Episode length', res.ep_length, timestamp=timestamp)
            exp.log_metric('Episode total return', np.sum(res.return_per_timestep), timestamp=timestamp)
            exp.log_metric('Execution time', res.execution_time)

            for i, count in enumerate(res.action_counts):
                exp.log_metric('Action {} count'.format(i), count, timestamp=timestamp)

    def log_action_reward(self, step, action, reward, episode_id):
        if self.enable_neptune:
            try:
                workspace = neptune.init(project_qualified_name=self.workspace_name)
                exp = workspace.get_experiments(id=episode_id)[0]
                exp.log_text("SAR", "Step {}, action {}, reward {}".format(step, action, reward))
                exp.log_metric("Return per timestep", reward)
            except Exception as e:
                print(e)

    def save_numpy(self, x, name="") -> None:
        assert self.numpy_dumps_dir is not None, \
            "[ERROR] create_directories must be called after initializing the logger and before starting to log"
        if name == "":
            name = "results"
        np.save(self.numpy_dumps_dir + "/" + name + ".npy", x)

    def save_prices(self, info_dict: dict, new_state, action: int, reward):
        save_path = info_dict['save_path']
        if bool(save_path):
            with open(save_path, 'a') as text_file:
                prices = ','.join(str(e) for e in new_state[:-1])
                # toprint = prices+','+str(a-1)+',real \n'
                toprint = prices + ',' + str(action - 1) + ',' + str(reward) + '\n'
                text_file.write(toprint)

    def plot_online_return(self, online_scores):
        plt.figure()
        plt.plot(online_scores)
        plt.grid = True
        plt.title("Return over policy improvement episodes")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.savefig(self.save_dir + "/plots/return.png")
        if not self.is_remote:
            plt.show()
        plt.close()

    def plot_loss(self, episode, ep_V_loss, ep_pi_loss):
        plt.figure()
        plt.plot(ep_V_loss, label="V_loss")
        plt.plot(ep_pi_loss, label="pi_loss")
        plt.grid = True
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.save_dir + "/plots/train_" + str(episode) + ".png")
        if not self.is_remote:
            plt.show()
        plt.close()

        ep_pi_loss = mean(ep_pi_loss)
        ep_V_loss = mean(ep_V_loss)

        self.training_V_loss.append(ep_V_loss)
        self.training_pi_loss.append(ep_pi_loss)

        print("--------------------------")
        print("Episode", episode)
        print("pi_loss:", ep_pi_loss)
        print("V_loss:", ep_V_loss)
        print("--------------------------")

    def plot_evaluation_mean_and_variance(self, avgs, stds, indexes=None):
        """Plot the mean and variance with a whiskers plot
        @type avgs: list
        @type stds: list
        @type indexes: list
        """
        if not indexes:
            indexes = [10 * i for i in range(len(avgs))]

        plt.figure()
        plt.errorbar(indexes, avgs, stds, linestyle='None', marker='^', capsize=3)
        plt.xlabel("Step of evaluation")
        plt.ylabel("Return")
        plt.title("Mean and variance for return in policy evaluation")
        if not self.is_remote:
            plt.show()
        plt.savefig(self.save_dir + "/plots/meanvariance.png")
        plt.close()

    def plot_training_loss_over_time(self):
        plt.figure()
        plt.plot(self.training_V_loss, label="V_loss")
        plt.plot(self.training_pi_loss, label="pi_loss")
        plt.grid = True
        plt.title("Loss over policy improvement episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.ylim = 3.0
        plt.legend()
        plt.savefig(self.save_dir + "/plots/overall.png")
        if not self.is_remote:
            plt.show()
        plt.close()

    def log_start(self, iteration, start_policy, start_value, start_targets):
        """Dump data about the starting game state over a txt file"""

        with open(self.save_dir+"/targets.txt", mode="a") as dump:
            dump.write("---- Targets at iteration " + str(iteration) + " ----\n")
            for target in start_targets:
                dump.write(str(target) + '\n')

            dump.write("---- Start policy ----\n")
            for n in start_policy:
                dump.write(str(n) + " ")
            dump.write("\n")

            dump.write("---- Start value ----\n")
            for n in start_value:
                dump.write(str(n) + " ")
            dump.write("\n")

            dump.close()

