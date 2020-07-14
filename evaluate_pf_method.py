#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
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

    # Disable running on GPU if not specifically requested
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
        game_params = {'horizon': args.max_ep_len}

        if args.game == 'Taxi' or args.game == 'TaxiEasy':
            game_params['grid'] = args.grid
            game_params['box'] = True
        elif args.game == 'RiverSwim-continuous':
            game_params['dim'] = args.chain_dim

        # particles = [5, 10, 25, 50, 100, 250, 500]

        # particles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        particles = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50]

        # Variables for storing experiments' results
        returns = []
        lens = []
        counts = []
        means = []
        stds = []
        indices = []
        rews = []
        gamma = args.gamma

        # Perform an experiment for each number of particles in the list
        for i in range(len(particles)):
            print()
            print("Number of particles:", particles[i])
            # n_mcts = int(args.budget/(args.max_ep_len * particles[i]))
            n_mcts = np.inf
            out_dir_i = out_dir + str(i) + '/'

            print("Number of mcts searches:", n_mcts)
            print("Maximum rollout length:", args.max_ep_len)
            episode_returns, timepoints, a_best, \
            seed_best, R_best, offline_scores = agent(game=args.game,
                                                      n_ep=args.n_ep,
                                                      n_mcts=n_mcts,
                                                      max_ep_len=args.max_ep_len,
                                                      budget=args.budget,
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
                                                      use_sampler=args.use_sampler,
                                                      unbiased=args.unbiased,
                                                      max_workers=args.max_workers,
                                                      variance=args.variance)

            total_rewards = offline_scores[0][0]
            returns_per_step = offline_scores[0][1]
            evaluation_lenghts = offline_scores[0][2]
            evaluation_pit_counts = offline_scores[0][3]

            means.append(np.mean(total_rewards))
            stds.append(2 * np.std(total_rewards) / np.sqrt(len(total_rewards)))  # 95% confidence interval
            for ret, length, count in zip(total_rewards, evaluation_lenghts, evaluation_pit_counts):
                returns.append(ret)
                lens.append(length)
                indices.append(str(particles[i]) + "_pf")
                counts.append(count)

            # Compute the discounted return
            for r_list in returns_per_step:
                discount = 1
                disc_rew = 0
                for r in r_list:
                    disc_rew += discount * r
                    discount *= gamma
                rews.append(disc_rew)

            # TODO FIX THIS
            # exps.append(offline_scores)
            # scores = np.stack(exps, axis=0)
            # np.save(out_dir + "scores.npy", scores)

        # Finished training: Visualize

        plt.figure()
        plt.errorbar(particles, means, stds, linestyle='None', marker='^', capsize=3)
        plt.xlabel("Number of particles")
        plt.ylabel("Undiscounted reward")
        plt.title("Particle filtering performance - budget {}".format(args.budget))
        plt.savefig(os.path.join(os.path.curdir, "logs/pf_evaluation_{}_{}.png".format(args.game, args.budget)))
        plt.close()

        # Store pandas dataframe with experiments' results
        data = {"agent": indices,
                "total_reward": returns,
                "discounted_reward": rews,
                "length": lens,
                "budget": [args.budget] * len(indices)}

        # Store the count of pit stops only if analyzing Race Strategy problem
        if "RaceStrategy" in args.game:
            data["pit_count"] = counts

        df = pd.DataFrame(data)

        df.to_csv("logs/" + time_str + '/' + "data_eval_pf_{}_{}.csv".format(args.game, args.budget), header=True,
                  index=False)
