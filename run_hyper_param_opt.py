import os, pickle
import numpy as np
import hyperopt as hp
from functools import partial

from hyperopt.mongoexp import MongoTrials

from agent import agent
from utils.parser_setup import setup_parser, parse_game_params
import time
results = []
base_dir = ''


def objective(params, keywords):
    # keywords["eval_freq"] = 1
    # keywords["n_ep"] = 1
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
    means = offline_scores[0][0]
    print("Mean return:", np.mean(means))
    print("Standard deviation:", np.std(means))
    results.append((params, np.mean(means), np.std(means), len(means)))
    return -np.mean(means)


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
                            "mid": args.mid}
    alg = "dpw/"
    if not args.stochastic:
        if args.unbiased:
            if args.variance:
                alg = 'p_uct_var/'
            else:
                alg = 'p_uct/'
        else:
            alg = 'pf_uct'
            if args.second_version:
                alg += '_2'
            alg += '/'
        #alg += str(args.particles) + '_particles/'
    start_time = time.time()
    time_str = str(start_time)
    out_dir = "logs/hyperopt/" + args.game + '/' + alg + str(args.budget) + "/" + time_str + '/'
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
            'second_version': args.second_version
    }

    # If a DB is available allocate accordingly the Trials object
    if args.db:
        trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key=args.dbname)
    else:
        trials = hp.Trials()

    if args.stochastic:
        parameter_space = {
            "c": hp.hp.quniform('c', 0.1, 3, 0.1),
            "alpha": hp.hp.quniform('alpha', 0, 0.99, 0.01)}
    else:
        parameter_space = {
            "c": hp.hp.quniform('c', 0.1, 3, 0.1)}
        if args.biased:
            parameter_space["alpha"] = hp.hp.quniform('alpha', args.min_alpha_hp, 0.99, 0.01)
    old = [{'alpha': 0.49, 'c': 1.6, 'temp': 0.05}, {'alpha': 0.99, 'c': 0.5, 'temp': 0.15}]

    best = hp.fmin(fn=partial(objective, keywords=keys), algo=hp.tpe.suggest, max_evals=args.opt_iters, space=parameter_space,
                   trials=trials) #, points_to_evaluate=old
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(hp.space_eval(parameter_space, best))

    # Save the trials file to resume
    if not args.db:
        with open(out_dir + "trials.pickle", "wb") as dumpfile:
            pickle.dump(trials, dumpfile)
            dumpfile.close()
        with open(out_dir + "results.pickle", "wb") as dumpfile:
            pickle.dump(results, dumpfile)
            dumpfile.close()