import os
import time
from copy import deepcopy
from functools import partial

from mango import Tuner, scheduler
import numpy as np

from agent import agent
from utils.parser_setup import parse_game_params, setup_parser, parse_alg_name
from scipy.stats import uniform

results = []
base_dir = ''


if __name__ == '__main__':

    args = setup_parser()

    out_dir = ""
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    game_params = parse_game_params(args)

    # Setup budget schedule parameters
    scheduler_params = None

    if args.budget_scheduler:
        assert args.min_budget < args.budget, "Minimum budget for the scheduler cannot be larger " \
                                              "than the overall budget"
        assert args.slope >= 1.0, "Slope lesser than 1 causes weird schedule function shapes"
        scheduler_params = {"slope": args.slope,
                            "min_budget": args.min_budget,
                            "min_depth": args.min_depth}
    alg = parse_alg_name(args)
    start_time = time.time()
    time_str = str(start_time)
    out_dir = "logs/mango/" + args.game + '/' + alg + str(args.budget)
    base_dir = out_dir
    keys = {"game": args.game,
            "n_ep": args.n_ep,
            "n_mcts": args.n_mcts,
            "max_ep_len": args.max_ep_len,
            "budget": args.budget,
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
            "n_workers": args.n_workers,
            "use_sampler": args.use_sampler,
            "particles": args.particles,
            "variance": args.variance,
            "unbiased": args.unbiased,
            "biased": args.biased,
            "depth_based_bias": args.depth_based_bias,
            "max_workers": args.max_workers,
            "scheduler_params": scheduler_params,
            'out_dir': out_dir,
            'second_version': args.second_version,
            'third_version': args.third_version,
            'csi': args.csi}


    # @scheduler.parallel(n_jobs=3)
    @scheduler.serial
    def objective(**params):
        keywords = deepcopy(keys)
        for k in params:
            keywords[k] = params[k]
        keywords['out_dir'] = base_dir + '/' + str(time.time()) + '/'
        print("Evaluating with params:", params)
        print("Saving results in:" + keywords['out_dir'])
        np.random.seed()
        _, _, _, _, _, offline_scores = agent(**keywords)
        means = []
        # Take all the average returns for evaluation
        # for score in offline_scores:
        #     means.append(score[2])
        means = offline_scores[0].episode_rewards
        print("Mean return:", np.mean(means))
        print("Standard deviation:", np.std(means))
        print()
        results.append((params, np.mean(means), np.std(means), len(means)))
        score = -np.mean(means)
        return score

    param_space = {"c": np.arange(0.1, 2.5, 0.1).tolist(),
                   #"csi": np.arange(0.25, 10.25, 0.25).tolist(),
                   "alpha": np.arange(0.4, 0.95, 0.05).tolist()}

    conf_dict = dict(num_iteration=args.opt_iters, initial_random=3)

    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.minimize()
    print('best parameters:', results['best_params'])
    print('best return:', results['best_objective'])