import argparse
from datetime import time

import numpy as np
import hyperopt as hp

from agent import agent

from functools import partial

parameter_space = {"lr": hp.hp.qloguniform('lr', np.log(0.0001), np.log(0.1)),
                   "temp": hp.hp.quniform('temp', 0.01, 0.1, 0.005)}


def objective(params, keywords):

    for k in params:
        keywords[k] = params[k]

    _, _, offline_scores = agent(**keywords)
    means = []

    # Take all the average returns for evaluation
    for score in offline_scores:
        means.append(score[2])

    return -np.mean(means)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01
                        , help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
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
    parser.add_argument('--eval_freq', type=int, default=20, help='Evaluation_frequency')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Episodes of evaluation')
    parser.add_argument('--delta_alpha', type=float, default=0.2, help='progressive widening parameter')
    parser.add_argument('--min_alpha', type=float, default=0, help='progressive widening parameter')
    parser.add_argument('--max_alpha', type=float, default=2, help='progressive widening parameter')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=16, help='Number of units per hidden layers in NN')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs of training for the NN')

    args = parser.parse_args()
    start_time = time.time()
    time_str = str(start_time)
    out_dir = 'logs/' + args.game + '/' + time_str + '/'

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    exps = []
    game_params = {}
    if args.game == 'Taxi' or args.game == 'TaxiEasy':
        game_params['grid'] = 'grid.txt'
        game_params['box'] = True
        # TODO modify this to return to original taxi problem
    for i in range(args.n_experiments):
        out_dir_i = out_dir + str(i) + '/'

        keys = {"game": args.game,
                "n_ep": args.n_ep,
                "n_mcts": args.n_mcts,
                "max_ep_len": args.max_ep_len,
                "lr": args.lr,
                "c": args.c,
                "gamma": args.gamma,
                "data_size": args.data_size,
                "batch_size": args.batch_size,
                "temp": args.temp,
                "n_hidden_layers": args.n_hidden_layers,
                "n_hidden_units": args.n_hidden_units,
                "stochastic": args.stochastic,
                "alpha": args.alpha,
                "numpy_dump_dir": out_dir_i,
                "visualize": args.visualize,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "pre_process": None,
                "game_params": game_params,
                "n_epochs": args.n_epochs}

        trials = hp.Trials()

        best = hp.fmin(fn=partial(objective, keys=keys), algo=hp.tpe.suggest, max_evals=10, space=parameter_space,
                       trials=trials)
        print(hp.space_eval(parameter_space, best))
