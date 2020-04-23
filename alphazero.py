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
import pandas as pd
import os
import matplotlib.pyplot as plt
from helpers import smooth, symmetric_remove
import time
from envs.blackjack_pi import BlackjackEnv

from utils.parser_setup import setup_parser

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

    # Obtain the command_line arguments
    args = setup_parser()

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
            print("Something wrong registering Blackjack environment")

    # Disable GPU acceleration if not specifically requested
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

        # Accept custom grid if the environment requires it
        if args.game == 'Taxi' or args.game == 'TaxiEasy':
            game_params['grid'] = args.grid
            game_params['box'] = True
            # TODO modify this to return to original taxi problem

        # Define the name of the agent to be stored in the dataframe
        if args.stochastic:
            agent_name = "dpw_"
        elif args.particles > 0:
            agent_name = str(args.particles) + "_pf_"
        else:
            agent_name = "classic_"

        if args.mcts_only:
            agent_name += "mcts_only"
        else:
            agent_name += "alphazero"

        # Run experiments
        for i in range(args.n_experiments):

            # Compute the actual number of mcts searches for each step
            if args.particles > 0:
                n_mcts = int(args.budget / (args.max_ep_len * args.particles))
            else:
                n_mcts = int(args.budget / args.max_ep_len)

            out_dir_i = out_dir + str(i) + '/'

            # Run the algorithm
            episode_returns, timepoints, a_best, \
            seed_best, R_best, offline_scores = agent(game=args.game,
                                                      n_ep=args.n_ep,
                                                      n_mcts=n_mcts,
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
                                                      particles=args.particles,
                                                      n_workers=args.n_workers,
                                                      use_sampler=args.use_sampler)

            total_rewards = offline_scores[0][0]
            undiscounted_returns = offline_scores[0][1]
            evaluation_lenghts = offline_scores[0][2]
            evaluation_terminal_states = offline_scores[0][3]

            indices = []
            returns = []
            lens = []
            rews = []

            gamma = args.gamma

            # Compute the discounted return
            for r_list in undiscounted_returns:
                discount = 1
                disc_rew = 0
                for r in r_list:
                    disc_rew += discount * r
                    discount *= gamma
                rews.append(disc_rew)

            # Fill the lists for building the dataframe
            for ret, length in zip(total_rewards, evaluation_lenghts):
                returns.append(ret)
                lens.append(length)
                indices.append(agent_name)

            # Store the result of the experiment
            data = {"agent": indices,
                    "total_reward": returns,
                    "discounted_reward": rews,
                    "length": lens,
                    "budget": [args.budget] * len(indices)}
            df = pd.DataFrame(data)
            df.to_csv("logs/{}_{}_{}_data_exp_{}.csv".format(agent_name, args.game, args.budget, i), header=True, index=False)

            # TODO FIX THIS
            # exps.append(offline_scores)
            # scores = np.stack(exps, axis=0)
            # np.save(out_dir + "scores.npy", scores)
