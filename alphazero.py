#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from utils.parser_setup import setup_parser, parse_game_params, parse_alg_name
plt.style.use('ggplot')
from agent import agent

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
    else:
        exps = []
        game_params = parse_game_params(args)

        # Define the name of the agent to be stored in the dataframe
        if args.stochastic:
            agent_name = "dpw_"
        elif args.model_based:
            agent_name = "model_based_"
        elif args.particles > 0:
            agent_name = str(args.particles) + "_pf_"
        else:
            agent_name = "classic_"

        if args.mcts_only:
            agent_name += "mcts_only"
        else:
            agent_name += "alphazero"

        # If required, prepare the budget scheduler parameters
        scheduler_params = None

        if args.budget_scheduler:
            assert args.min_budget < args.budget, "Minimum budget for the scheduler cannot be larger " \
                                                  "than the overall budget"
            assert args.slope >= 1.0, "Slope lesser than 1 causes weird schedule function shapes"
            scheduler_params = {"slope": args.slope,
                                "min_budget": args.min_budget,
                                "mid": args.mid}

        # Run experiments
        for i in range(args.n_experiments):

            n_mcts = np.inf

            out_dir_i = out_dir + str(i) + '/'
            alg = parse_alg_name(args)

            out_dir = "logs/" + args.game
            if args.game == 'RiverSwim-continuous':
                out_dir += "/" + "fail_" + str(args.fail_prob)
            out_dir += "/" + alg + time_str + '/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # Run the algorithm
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
                                                      pre_process=None,
                                                      game_params=game_params,
                                                      n_epochs=args.n_epochs,
                                                      parallelize_evaluation=args.parallel,
                                                      mcts_only=args.mcts_only,
                                                      particles=args.particles,
                                                      n_workers=args.n_workers,
                                                      use_sampler=args.use_sampler,
                                                      unbiased=args.unbiased,
                                                      biased=args.biased,
                                                      variance=args.variance,
                                                      depth_based_bias=args.depth_based_bias,
                                                      max_workers=args.max_workers,
                                                      scheduler_params=scheduler_params,
                                                      out_dir=out_dir,
                                                      render=args.render,
                                                      second_version=args.second_version,
                                                      third_version=args.third_version,
                                                      model_based=args.model_based,
                                                      on_visits=args.on_visits)

            total_rewards = offline_scores[0][0]
            undiscounted_returns = offline_scores[0][1]
            evaluation_lenghts = offline_scores[0][2]
            evaluation_pit_action_counts = offline_scores[0][3]

            indices = []
            returns = []
            lens = []
            rews = []
            counts = []

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
            for ret, length, count in zip(total_rewards, evaluation_lenghts, evaluation_pit_action_counts):
                returns.append(ret)
                lens.append(length)
                indices.append(agent_name)
                counts.append(count)

            # Store the result of the experiment
            data = {"agent": indices,
                    "total_reward": returns,
                    "discounted_reward": rews,
                    "length": lens,
                    "budget": [args.budget] * len(indices)}

            # Store the count of pit stops only if analyzing Race Strategy problem
            if "RaceStrategy" in args.game:
                data["pit_count"] = counts

            # Write the dataframe to csv
            df = pd.DataFrame(data)
            df.to_csv(out_dir + "{}_{}_{}_data_exp_{}.csv".format(agent_name, args.game, args.budget
                                                                                   , i), header=True, index=False)

            # TODO FIX THIS
            # exps.append(offline_scores)
            # scores = np.stack(exps, axis=0)
            # np.save(out_dir + "scores.npy", scores)
