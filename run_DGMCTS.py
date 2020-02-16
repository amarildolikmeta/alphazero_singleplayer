#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from agents.mcts_agent import MCTSAgent
from envs.taxi import generate_taxi
from envs.gridworld import generate_gridworld
from envs.chain import generate_chain
from envs.loop import generate_loop
from envs.river_swim import generate_river
from envs.six_arms import generate_arms
from envs.three_arms import generate_arms as generate_three_arms
from rl.make_game import make_game
from envs.blackjack_pi import BlackjackEnv
register(
    id='Blackjack_pi-v0',
    entry_point='envs.blackjack_pi:BlackjackEnv',
)

#### Command line call, parsing and plotting ##
colors = ['r', 'b', 'g', 'orange', 'c', 'k', 'purple', 'y']
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Chain', help='Training environment',
                        choices=[
                            "Gridworld",
                            "Chain",
                            "Taxi",
                            "Loop",
                            "RiverSwim",
                            "SixArms",
                            "ThreeArms",
                            "Blackjack"
                            ""],)
    parser.add_argument('--n_ep', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_states', type=int, default=5, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=30, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=100, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=0.9999, help='Discount parameter')
    parser.add_argument('--alpha', type=float, default=0.6, help='progressive widening parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--alpha_test', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=20, help='Evaluation_frequency')
    parser.add_argument('--eval_episodes', type=int, default=20, help='Episodes of evaluation')
    parser.add_argument('--delta_alpha', type=float, default=0.2, help='progressive widening parameter')
    parser.add_argument('--min_alpha', type=float, default=0, help='progressive widening parameter')
    parser.add_argument('--max_alpha', type=float, default=2, help='progressive widening parameter')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=16, help='Number of units per hidden layers in NN')

    args = parser.parse_args()
    start_time = time.time()
    time_str = str(start_time)
    out_dir = 'logs/' + args.game + '/' + time_str + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    name = args.game
    if name == 'Taxi':
        mdp = generate_taxi('./grid.txt', horizon=5000, gamma=args.gamma, prob=0.5)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 5000
    elif name == 'Chain':
        mdp = generate_chain(horizon=100, gamma=args.gamma, n=args.n_states, large=100)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'Gridworld':
        mdp = generate_gridworld(horizon=100, gamma=args.gamma)
        max_steps = 500000
        evaluation_frequency = 5000
        test_samples = 1000
    elif name == 'Loop':
        mdp = generate_loop(horizon=100, gamma=args.gamma)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'RiverSwim':
        mdp = generate_river(horizon=100, gamma=args.gamma)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
        mbie_C = 0.4
    elif name == 'SixArms':
        mdp = generate_arms(horizon=100, gamma=args.gamma)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
        mbie_C = 0.8
    elif name == 'ThreeArms':
        horizon = 100
        mdp = generate_three_arms(horizon=horizon, gamma=args.gamma)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    elif name == 'Blackjack':
        horizon = 100
        mdp = BlackjackEnv(box=False)
        max_steps = 100000
        evaluation_frequency = 1000
        test_samples = 1000
    agent = MCTSAgent(ns=mdp.info.size[0], na=mdp.info.size[1], gamma=args.gamma)
    def pre_process():
        from gym.envs.registration import register
        try:
            register(
                id='Blackjack_pi-v0',
                entry_point='envs.blackjack_pi:BlackjackEnv',
            )
        except:
            pass

    exps = []
    for i in range(args.n_experiments):
        episode_returns, timepoints, R_best, offline_scores = agent.learn(env=mdp,
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
                                                      alpha=args.alpha,
                                                      out_dir=out_dir,
                                                      visualize=args.visualize,
                                                      eval_freq=args.eval_freq,
                                                      eval_episodes=args.eval_episodes,
                                                      pre_process=None)
        exps.append(offline_scores)
        scores = np.stack(exps, axis=0)
        np.save("scores.npy", scores)

    # Finished training: Visualize
    # fig, ax = plt.subplots(1, figsize=[7, 5])
    # total_eps = len(episode_returns)
    # episode_returns = smooth(episode_returns, args.window, mode='valid')
    # ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1), episode_returns,  color='darkred')
    # ax.set_ylabel('Return')
    # ax.set_xlabel('Episode', color='darkred')
    # name = 'learning_curve' + ('_dpw_alpha_'+str(args.alpha) if args.stochastic else '') + '.png'
    # plt.savefig(out_dir + name, bbox_inches="tight")
