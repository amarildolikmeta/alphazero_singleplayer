import argparse
import os

import numpy as np
import hyperopt as hp

from agent import agent

from functools import partial

parameter_space = {"temp": hp.hp.quniform('temp', 0.05, 0.5, 0.05),
                   # "lr": hp.hp.qloguniform('lr', np.log(0.0001), np.log(0.1)),
                   "c": hp.hp.quniform('c', 0, 2, 0.1),
                   "alpha": hp.hp.quniform('alpha', 0, 0.99, 0.01)}

import pickle


def objective(params, keywords):

    print(params)

    # keywords["eval_freq"] = 1
    # keywords["n_ep"] = 1
    for k in params:
        keywords[k] = params[k]

    _, _, _, _, _, offline_scores = agent(**keywords)
    means = []

    # Take all the average returns for evaluation
    for score in offline_scores:
        means.append(score[2])

    print("Mean return:", -np.mean(means))
    print("Standard deviation:", np.std(means))

    return -np.mean(means)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--grid', type=str, default="grid.txt", help='TXT file specfying the game grid')
    parser.add_argument('--n_ep', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01
                        , help='Learning rate')
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

    args = parser.parse_args()
    out_dir = ""
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    game_params = {}
    if args.game == 'Taxi' or args.game == 'TaxiEasy':
        game_params['grid'] = args.grid
        game_params['box'] = True
        # TODO modify this to return to original taxi problem

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
            "numpy_dump_dir": out_dir,
            "visualize": args.visualize,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "pre_process": None,
            "game_params": game_params,
            "n_epochs": args.n_epochs,
            "parallelize_evaluation": args.parallel,
            "mcts_only": args.mcts_only
    }

    trials = hp.Trials()

    old = [{'alpha': 0.49, 'c': 1.6, 'temp': 0.05}, {'alpha': 0.99, 'c': 0.5, 'temp': 0.15}]

    best = hp.fmin(fn=partial(objective, keywords=keys), algo=hp.tpe.suggest, max_evals=50, space=parameter_space,
                   trials=trials, points_to_evaluate=old)
    print(hp.space_eval(parameter_space, best))

    with open("trials.pickle", "wb") as dumpfile:
        pickle.dump(trials, dumpfile)
        dumpfile.close()
