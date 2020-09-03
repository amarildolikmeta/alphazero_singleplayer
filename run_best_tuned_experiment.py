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
from utils.parser_setup import setup_parser
import pickle
import glob
plt.style.use('ggplot')
from agent import agent

#### Command line call, parsing and plotting ##
colors = ['r', 'b', 'g', 'orange', 'c', 'k', 'purple', 'y']
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']
envs = [ 'Trading-v0', 'RiverSwim-continuous',]
budgets = [1000, 5000, 10000, 20000]
settings = ['dpw', 'p_uct', 'pf_uct', ]
setting_to_sub = {
    'dpw': '',
    'p_uct': '1_particles/',
    'pf_uct': '1_particles/',
    'pf_uct_2':'1_particles/',
}
setting_to_agent = {
    'dpw': 'dpw',
    'p_uct': '1_pf_mcts_only',
    'pf_uct': '1_pf_mcts_only',
    'pf_uct_2': '1_pf_mcts_only'
}
setting_to_label = {
    'dpw': 'dpw',
    'p_uct': 'ol_uct',
    'pf_uct': 'pf_uct',
    'pf_uct_2': 'pf_uct_2'
}
env_to_sub = {
    'RaceStrategy':'',
    'RiverSwim-continuous': '',
    'Trading-v0':'',
    'Gridworld':'',
    'Cliff':'',
}
env_to_label = {
    'RaceStrategy': 'RaceStrategy',
    'RiverSwim-continuous': 'Riverswim',
    'Trading-v0': 'Trading',
    'Gridworld': 'Gridworld',
    'Cliff': 'Cliff',
}
if __name__ == '__main__':

    # Obtain the command_line arguments
    args = setup_parser()
    out_dir = 'logs/best_tune/'

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

    for e, env in enumerate(envs):
        sub_env = env_to_sub[env]
        env_label = env_to_label[env]
        for i, setting in enumerate(settings):
            sub = setting_to_sub[setting]
            agent_label = setting_to_agent[setting]
            for budget in budgets:
                path = "logs/hyperopt/{}/{}{}/{}/*/results.pickle".format(env,sub_env, setting, str(budget))
                paths = glob.glob(path)
                data = []
                if len(paths) == 0:
                    print("No path found " + setting + "-" + str(budget))
                    continue
                results = []
                for p in paths:
                    with open(p, 'rb') as f:
                        results += pickle.load(f)
                res = [x[1] for x in results]
                best_params = results[np.argmax(res)][0]
                start_time = time.time()
                time_str = str(start_time)
                out_dir += env + '/' + time_str + '/'
                game_params = {'horizon': args.max_ep_len}

                # Accept custom grid if the environment requires it
                if env == 'Taxi' or env == 'TaxiEasy':
                    game_params['grid'] = args.grid
                    game_params['box'] = True
                    max_ep_len = 20
                    # TODO modify this to return to original taxi problem
                elif env == 'RiverSwim-continuous':
                    game_params['dim'] = 7
                    game_params['fail'] = 0.1
                    max_ep_len = 20
                elif env == 'RaceStrategy':
                    game_params['scale_reward'] = True
                    max_ep_len = 20
                elif env == 'Trading-v0':
                    max_ep_len = 50
                else:
                    max_ep_len = 20

                fun_args = [env, args.n_ep, args.n_mcts, max_ep_len, args.lr, args.c, args.gamma,
                            args.data_size, args.batch_size, args.temp, args.n_hidden_layers, args.n_hidden_units,
                            True, args.eval_freq, args.eval_episodes, args.n_epochs]
                exps = []

                c = best_params['c']
                # Define the name of the agent to be stored in the dataframe
                if setting == 'dpw':
                    agent_name = "dpw_"
                    stochastic = True
                    particles = 0
                    biased = unbiased = False
                    alpha = best_params['alpha']
                elif setting in ['p_uct', 'pf_uct']:
                    stochastic = False
                    particles = 1
                    if setting in ['p_uct']:
                        unbiased = True
                        biased = False
                        alpha = 1.
                    else:
                        unbiased = False
                        biased = True
                        alpha = best_params['alpha']
                    agent_name = str(particles) + "_pf_"
                else:
                    stochastic = False
                    particles = 0
                    biased = unbiased = False
                    agent_name = "classic_"

                if args.mcts_only:
                    agent_name += "mcts_only"
                else:
                    agent_name += "alphazero"

                # If required, prepare the budget scheduler parameters
                scheduler_params = None
                print("Performing experiment with budget " + str(budget) + "!")
                if args.budget_scheduler:
                    assert args.min_budget < budget, "Minimum budget for the scheduler cannot be larger " \
                                                          "than the overall budget"
                    assert args.slope >= 1.0, "Slope lesser than 1 causes weird schedule function shapes"
                    scheduler_params = {"slope": args.slope,
                                        "min_budget": args.min_budget,
                                        "mid": args.mid}
                alg = "dpw/"
                if not stochastic:
                    if unbiased:
                        if args.variance:
                            alg = 'p_uct_var/'
                        else:
                            alg = 'p_uct/'
                    else:
                        alg = 'pf_uct/'
                out_dir = "logs/tuned_exp/" + env
                if env == 'RiverSwim-continuous':
                    out_dir += "/" + "fail_" + str(0.1)
                out_dir += "/" + alg + str(budget) + '/' + time_str + '/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # Run experiments
                n_mcts = np.inf
                out_dir_i = out_dir + '/'
                # Run the algorithm
                episode_returns, timepoints, a_best, \
                seed_best, R_best, offline_scores = agent(game=env,
                                                          n_ep=args.n_ep,
                                                          n_mcts=n_mcts,
                                                          max_ep_len=max_ep_len,
                                                          budget=budget,
                                                          lr=args.lr,
                                                          c=c,
                                                          gamma=args.gamma,
                                                          data_size=args.data_size,
                                                          batch_size=args.batch_size,
                                                          temp=args.temp,
                                                          n_hidden_layers=args.n_hidden_layers,
                                                          n_hidden_units=args.n_hidden_units,
                                                          stochastic=stochastic,
                                                          alpha=alpha,
                                                          numpy_dump_dir=out_dir_i,
                                                          visualize=False,
                                                          eval_freq=args.eval_freq,
                                                          eval_episodes=args.eval_episodes,
                                                          pre_process=None,
                                                          game_params=game_params,
                                                          n_epochs=args.n_epochs,
                                                          parallelize_evaluation=args.parallel,
                                                          mcts_only=args.mcts_only,
                                                          particles=particles,
                                                          n_workers=args.n_workers,
                                                          use_sampler=args.use_sampler,
                                                          unbiased=unbiased,
                                                          biased=biased,
                                                          variance=args.variance,
                                                          depth_based_bias=args.depth_based_bias,
                                                          max_workers=args.max_workers,
                                                          scheduler_params=scheduler_params,
                                                          out_dir=out_dir)

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
                        "budget": [budget] * len(indices)}

                # Store the count of pit stops only if analyzing Race Strategy problem
                if "RaceStrategy" in env:
                    data["pit_count"] = counts

                # Write the dataframe to csv
                df = pd.DataFrame(data)
                df.to_csv(out_dir + "/data.csv", header=True, index=False)
