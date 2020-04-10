#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
import errno
import json
from datetime import datetime

from joblib import Parallel, delayed
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from helpers import smooth, symmetric_remove
import time
from envs.blackjack_pi import BlackjackEnv

plt.style.use('ggplot')
from agent import agent
from gym.envs.registration import register

register(
    id='Blackjack_pi-v0',
    entry_point='envs.blackjack_pi:BlackjackEnv',
)

#### Command line call, parsing and plotting ##
colors = ['r', 'b', 'g', 'orange', 'c', 'k', 'purple', 'y']
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--grid', type=str, default="grid.txt", help='TXT file specfying the game grid')
    parser.add_argument('--n_ep', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--cdpw', type=float, default=1, help='DPW constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--alpha', type=float, default=0.6, help='progressive widening parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--alpha_test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=20, help='Evaluation_frequency')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Episodes of evaluation')
    parser.add_argument('--delta_alpha', type=float, default=0.2, help='progressive widening parameter')
    parser.add_argument('--min_alpha', type=float, default=0, help='progressive widening parameter')
    parser.add_argument('--max_alpha', type=float, default=2, help='progressive widening parameter')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=16, help='Number of units per hidden layers in NN')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs of training for the NN')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--mcts_only', action='store_true')
    parser.add_argument('--show_plots', action='store_true')
    parser.add_argument('--particles', type=int, default=0, help='Numbers of particles to approximate state distributions')

    args = parser.parse_args()
    start_time = time.time()
    time_str = str(start_time)
    out_dir = 'logs/' + args.game + '/' + time_str + '/'

    def pre_process():
        from gym.envs.registration import register
        try:
            register(
                id='Blackjack_pi-v0',
                entry_point='envs.blackjack_pi:BlackjackEnv',
            )
        except:
            pass


    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    fun_args = [args.game, args.n_ep, args.n_mcts, args.max_ep_len, args.lr, args.c, args.gamma,
                args.data_size, args.batch_size, args.temp, args.n_hidden_layers, args.n_hidden_units,
                True, args.eval_freq, args.eval_episodes, args.n_epochs]
    if args.alpha_test:
        alpha = args.min_alpha
        delta_alpha = args.delta_alpha
        n = int((args.max_alpha - args.min_alpha) / delta_alpha)
        affinity = min(min(len(os.sched_getaffinity(0)), n), 4)
        out = Parallel(n_jobs=affinity)(
            delayed(agent)(*(fun_args + [alpha + i * delta_alpha, out_dir + '/alpha_' +
                                         str(alpha + i * delta_alpha) + '/', pre_process]))
            for i in range(n))

        # fig, ax = plt.subplots(1, figsize=[7, 5])
        # for j in range(len(out)):
        #     episode_returns, timepoints, a_best, seed_best, R_best, offline_scores = out[j]
        #     total_eps = len(episode_returns)
        #     episode_returns = smooth(episode_returns, args.window, mode='valid')
        #     ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1), episode_returns, color=colors[j],
        #         marker =markers[j], label='a-'+ str(alpha + j * delta_alpha))
        #     model.save(out_dir+'/alpha_' + str(alpha + j * delta_alpha) + '/')
        # ax.set_ylabel('Return')
        # ax.set_xlabel('Episode', color='darkred')
        # name = 'learning_curve_dpw_alpha_test.png'
        # lgd = fig.legend(loc='lower center', ncol=n/2, fancybox=True, shadow=True)
        # plt.savefig(out_dir + name, bbox_inches="tight", bbox_extra_artists=(lgd,))
    else:
        exps = []
        game_params = {}

        if args.game == 'Taxi' or args.game == 'TaxiEasy':
            game_params['grid'] = args.grid
            game_params['box'] = True

        # particles = [5, 10, 25, 50, 100, 250, 500]

        particles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        returns = []
        lens = []
        final_states = []
        means = []
        stds = []

        for i in range(len(particles)):
            print()
            print("Number of particles:", particles[i])
            out_dir_i = out_dir + str(i) + '/'
            episode_returns, timepoints, a_best, \
            seed_best, R_best, offline_scores = agent(game=args.game,
                                                      n_ep=args.n_ep,
                                                      n_mcts=args.n_mcts,
                                                      max_ep_len=args.max_ep_len,
                                                      lr=args.lr,
                                                      c=args.c,
                                                      gamma=args.gamma,
                                                      data_size=args.data_size,
                                                      batch_size=args.batch_size,
                                                      temp=args.temp,
                                                      n_hidden_layers=args.n_hidden_layers,
                                                      n_hidden_units=args.n_hidden_units,
                                                      stochastic=args.stochastic,
                                                      alpha=args.alpha,
                                                      numpy_dump_dir=out_dir_i,
                                                      visualize=args.visualize,
                                                      eval_freq=args.eval_freq,
                                                      eval_episodes=args.eval_episodes,
                                                      pre_process=None, game_params=game_params,
                                                      n_epochs=args.n_epochs,
                                                      parallelize_evaluation=args.parallel,
                                                      mcts_only=args.mcts_only,
                                                      particles=particles[i],
                                                      n_workers=args.n_workers,
                                                      use_sampler=args.use_sampler)

            evaluation_returns = offline_scores[0][0]
            evaluation_lenghts = offline_scores[0][1]
            evaluation_terminal_states = offline_scores[0][2]

            means.append(np.mean(evaluation_returns))
            stds.append(2 * np.std(evaluation_returns) / np.sqrt(len(evaluation_returns)))  # 95% confidence interval
            returns.append(evaluation_returns)
            lens.append(evaluation_lenghts)
            final_states.append(evaluation_terminal_states)

            # TODO FIX THIS
            # exps.append(offline_scores)
            # scores = np.stack(exps, axis=0)
            # np.save(out_dir + "scores.npy", scores)

        # Finished training: Visualize

        plt.figure()
        plt.errorbar(particles, means, stds, linestyle='None', marker='^', capsize=3)
        plt.xlabel("Number of particles")
        plt.ylabel("Return")
        plt.title("Mean and variance for number of particles")
        plt.savefig(os.path.join(os.path.curdir, "logs/pf_evaluation.png"))
        plt.close()

        print("Averages:", means)
        print("Standard deviations:", stds)

        with open("logs/stats.txt", 'w') as out:
            for i in range(len(particles)):
                out.write(str(particles[i]) + " particles\n")
                out.write("Returns: " + str(returns[i])+"\n")
                out.write("Mean: " + str(means[i]) + ", Std: " + str(stds[i]) + "\n")
                out.write("Episode durations: " + str(lens[i]) + "\n")
                out.write("Episode final states:\n" + str(final_states[i]) + "\n\n")


        # fig, ax = plt.subplots(1, figsize=[7, 5])
        # total_eps = len(episode_returns)
        # episode_returns = smooth(episode_returns, args.window, mode='valid')
        # ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1), episode_returns,  color='darkred')
        # ax.set_ylabel('Return')
        # ax.set_xlabel('Episode', color='darkred')
        # name = 'learning_curve' + ('_dpw_alpha_'+str(args.alpha) if args.stochastic else '') + '.png'
        # plt.savefig(out_dir + name, bbox_inches="tight")