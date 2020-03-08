import errno
import json
import os
from datetime import datetime
from statistics import mean

from matplotlib import pyplot as plt

class Logger(object):
    def __init__(self, params, game, remote=True):
        self.save_dir, self.numpy_dumps_dir = self.save_parameters(params, game)
        self.is_remote = remote
        self.training_V_loss = []
        self.training_pi_loss = []

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
        plt.savefig(self.save_dir + "/plots//train_" + str(episode) + ".png")
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

    def plot_evaluation_mean_and_variance(self, avgs, stds):
        """Plot the mean and variance with a whiskers plot
        @type avgs: list
        @type stds: list
        """
        plt.figure()
        plt.errorbar([10 * i for i in range(len(avgs))], avgs, stds, linestyle='None', marker='^', capsize=3)
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
        with open(self.save_dir+"/targets.txt", mode="a") as dump:
            dump.write("---- Targets at iteration " + str(iteration) + " ----\n")
            for target in start_targets:
                dump.write(str(target) + '\n')

            dump.write("---- Start policy ----\n")
            for n in start_policy.squeeze():
                dump.write(str(n) + " ")
            dump.write("\n")

            dump.write("---- Start value ----\n")
            for n in start_value.squeeze():
                dump.write(str(n) + " ")
            dump.write("\n")

    @staticmethod
    def save_parameters(params, game):
        mydir = os.path.join(
            os.getcwd(), "logs", game,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.makedirs(mydir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

        try:
            os.makedirs(os.path.join(mydir, "plots"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

        try:
            os.makedirs(os.path.join(mydir, "numpy_dumps"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..

        with open(os.path.join(mydir, "parameters.txt"), 'w') as d:
            d.write(json.dumps(params))

        return mydir, os.path.join(mydir, "numpy_dumps")