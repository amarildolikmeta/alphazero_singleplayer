import os
import time
from copy import deepcopy
from functools import partial

from mango import Tuner, scheduler
import numpy as np

from agent import agent
from utils.parser_setup import parse_game_params, setup_parser, parse_alg_name
import pickle


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
    #TODO add missing arguments -> standardize?
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
            'csi': args.csi,
            'bayesian': args.bayesian,
            'q_learning': args.q_learning,
            'ucth': args.ucth,
            'log_timestamp': time_str,
            'verbose': args.verbose,
            'power': args.power,
            'p': args.p,
            'beta': args.beta}


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

        # The optimizer minimizes the score, ensure it is always negative to obtain a maximization
        mean_score = np.mean(means)
        if mean_score < 0:
            mean_score = -mean_score
        with open(os.path.join(out_dir, "results.pickle"), 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
        return mean_score

    param_space = {"c": np.arange(50, 160, 10).tolist(),
                   "alpha": np.arange(0.1, 3.1, 0.1).tolist(),
                   "beta": np.arange(1, 10, 0.5).tolist()}
                   # "alpha": np.arange(0.4, 0.95, 0.05).tolist()}

    conf_dict = dict(num_iteration=args.opt_iters, initial_random=3)

    tuner = Tuner(param_space, objective, conf_dict)
    final_result = tuner.minimize()
    print('best parameters:', final_result['best_params'])
    print('best return:', final_result['best_objective'])

    # TODO save the parameters in a log folder