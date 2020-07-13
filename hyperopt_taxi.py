import argparse
import os, pickle
import numpy as np
import hyperopt as hp
from functools import partial

from agent import agent
from utils.parser_setup import setup_parser

parameter_space = {"temp": hp.hp.quniform('temp', 0.05, 0.5, 0.05),
                   # "lr": hp.hp.qloguniform('lr', np.log(0.0001), np.log(0.1)),
                   "c": hp.hp.quniform('c', 0, 2, 0.1),
                   "alpha": hp.hp.quniform('alpha', 0, 0.99, 0.01)}


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

    args = setup_parser()

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
            "mcts_only": args.mcts_only,
            "particles": args.particles,
            "variance": args.variance
    }

    trials = hp.Trials()

    old = [{'alpha': 0.49, 'c': 1.6, 'temp': 0.05}, {'alpha': 0.99, 'c': 0.5, 'temp': 0.15}]

    best = hp.fmin(fn=partial(objective, keywords=keys), algo=hp.tpe.suggest, max_evals=50, space=parameter_space,
                   trials=trials, points_to_evaluate=old)
    print(hp.space_eval(parameter_space, best))

    with open("trials.pickle", "wb") as dumpfile:
        pickle.dump(trials, dumpfile)
        dumpfile.close()
