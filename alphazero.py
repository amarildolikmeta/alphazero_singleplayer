#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
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
colors = ['r', 'b', 'g','orange','c', 'k','purple', 'y']
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^','2','1','3','4']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=100, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--alpha', type=float, default=0.6, help='progressive widening parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--alpha_test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation_frequency')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Episodes of evaluation')
    parser.add_argument('--delta_alpha', type=float, default=0.2, help='progressive widening parameter')
    parser.add_argument('--min_alpha', type=float, default=0, help='progressive widening parameter')
    parser.add_argument('--max_alpha', type=float, default=2,  help='progressive widening parameter')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=16, help='Number of units per hidden layers in NN')

    args = parser.parse_args()
    start_time = time.time()
    time_str = str(start_time)
    out_dir = 'logs/'+args.game+'/'+time_str+'/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fun_args = [args.game, args.n_ep, args.n_mcts, args.max_ep_len, args.lr, args.c, args.gamma,
                args.data_size, args.batch_size,args.temp, args.n_hidden_layers, args.n_hidden_units,
                True, args.eval_freq, args.eval_episodes]
    if args.alpha_test:
        alpha = args.min_alpha
        delta_alpha = args.delta_alpha
        n = int((args.max_alpha - args.min_alpha) / delta_alpha)
        affinity = min(len(os.sched_getaffinity(0)), n)
        out = Parallel(n_jobs=affinity)(
            delayed(agent)(*(fun_args + [ alpha + i * delta_alpha, out_dir+'/alpha_' +
                                          str(alpha + i * delta_alpha) + '/']))
            for i in range(n))

        fig, ax = plt.subplots(1, figsize=[7, 5])
        # for j in range(len(out)):
        #     episode_returns, timepoints, a_best, seed_best, R_best, model = out[j]
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
        episode_returns, timepoints, a_best, seed_best, R_best, model = agent(game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts,
                                                                       max_ep_len=args.max_ep_len, lr=args.lr, c=args.c,
                                                                       gamma=args.gamma,
                                                                       data_size=args.data_size,
                                                                       batch_size=args.batch_size,
                                                                       temp=args.temp,
                                                                       n_hidden_layers=args.n_hidden_layers,
                                                                       n_hidden_units=args.n_hidden_units,
                                                                       stochastic=args.stochastic,
                                                                       alpha=args.alpha,
                                                                       out_dir=out_dir,
                                                                       visualize=args.visualize,
                                                                       eval_freq=args.eval_freq,
                                                                       eval_episodes=args.eval_episodes)

        # Finished training: Visualize
        fig, ax = plt.subplots(1, figsize=[7, 5])
        total_eps = len(episode_returns)
        episode_returns = smooth(episode_returns, args.window, mode='valid')
        ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1), episode_returns,  color='darkred')
        ax.set_ylabel('Return')
        ax.set_xlabel('Episode', color='darkred')
        name = 'learning_curve' + ('_dpw_alpha_'+str(args.alpha) if args.stochastic else '') + '.png'
        plt.savefig(out_dir + name, bbox_inches="tight")

#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
