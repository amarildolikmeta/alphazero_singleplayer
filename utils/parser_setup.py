import argparse


def parse_game_params(args):
    game_params = {'horizon': args.max_ep_len}
    # Accept custom grid if the environment requires it
    if args.game in ['Trading-v0', 'Trading_discrete-v0']:
        game_params['max_ret'] = args.max_ret
        game_params['n_ret'] = args.n_ret
        game_params['fees'] = args.fees
    if args.game == 'Taxi' or args.game == 'TaxiEasy':
        game_params['grid'] = args.grid
        game_params['box'] = args.box
    if args.game == 'GridworldDiscrete':
        game_params['scale_reward'] = args.scale_reward
        game_params['fail_prob'] = args.fail_prob
        # TODO modify this to return to original taxi problem
    elif args.game in ['RiverSwim-continuous', 'MountainCar', 'Cartpole']:
        game_params['fail_prob'] = args.fail_prob
        if args.game in ['RiverSwim-continuous']:
            game_params['dim'] = args.chain_dim
    elif args.game == 'RaceStrategy':
        game_params['scale_reward'] = args.scale_reward
    elif args.game in ['RaceStrategy-full', 'RaceStrategy-v1']:
        game_params['scale_reward'] = args.scale_reward
        game_params['n_cores'] = args.max_xgb_workers
    return game_params



def parse_alg_name(args):
    alg = "dpw/"
    if args.model_based:
        alg = 'model_based/'
    elif not args.stochastic:
        if args.unbiased:
            if args.variance:
                alg = 'p_uct_var/'
            else:
                alg = 'p_uct/'
        else:
            alg = 'pf_uct'
            if args.second_version:
                alg += '_2'
            elif args.third_version:
                alg += '_3'
            alg += '/'
    return alg

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--grid', type=str, default="grid.txt", help='TXT file specfying the game grid')
    parser.add_argument('--n_ep', type=int, default=1, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--chain_dim', type=int, default=10, help='Chain dimension')
    parser.add_argument('--fail_prob', type=float, default=0.1, help='Fail probability in riverswim')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--cdpw', type=float, default=1, help='DPW constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--alpha', type=float, default=0.6, help='progressive widening parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--n_experiments', type=int, default=1, help='Number of experiments')
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
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--mcts_only', action='store_true')
    parser.add_argument('--show_plots', action='store_true')
    parser.add_argument('--particles', type=int, default=0,
                        help='Numbers of particles to approximate state distributions')
    parser.add_argument('--unbiased', action='store_true', help='Use the unbiased particle algorithm')
    parser.add_argument('--biased', action='store_true', help='Use the biased particle algorithm')
    parser.add_argument('--variance', action='store_true', help='use variance based selection policy')
    parser.add_argument('--max_workers', type=int, default=100, help='Maximum number of parallel workers')
    parser.add_argument('--budget', type=int, default=1000, help='Computational budget')
    parser.add_argument('--depth_based_bias', action='store_true', help='use depth based bias')

    # Budget scheduler args
    parser.add_argument('--budget_scheduler', action='store_true', help='Enable budget scheduler')
    parser.add_argument('--slope', type=float, default=1.0, help='Constant regulating the slope for the scheduler '
                                                                 'decay')
    parser.add_argument('--min_budget', type=int, default=1, help='Minimum budget value to be returned by the '
                                                                  'scheduler')

    # Hyperparameter optimization args
    parser.add_argument('--opt_iters', type=int, default=20, help='Number of hyperparameter tries,'
                                                                  ' only used in hyperparameter tuning')
    parser.add_argument('--mid', type=float, default=0.0, help='Constant regulating the middle point of the slope')
    parser.add_argument('--db', action='store_true', help="Use MongoDB parallelization for hyperparameters tuning")
    parser.add_argument('--dbname', type=str, default="exp_0", help="Name of the db to be used for the experiment")
    parser.add_argument('--min_alpha_hp', type=float, default=0.5, help='Minimum alpha in hyperopt for pf_uct')

    parser.add_argument('--scale_reward', action='store_true', help='scale the reward of the race environment')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--second_version', action='store_true', help='only for pf_uct2')
    parser.add_argument('--third_version', action='store_true', help='only for pf_uct3')

    # Race strategy model arguments
    parser.add_argument('--max_xgb_workers', type=int, default=1, help="Number of threads to be used by each XGB model,"
                                                                        "only used for RaceStrategy environment")
    parser.add_argument('--model_based', action='store_true', help='Use the model_based algorithm')
    parser.add_argument('--box', action='store_true', help='Used for Taxi environment')
    parser.add_argument('--on_visits', action='store_true', help='Make final selection based on action counts')

    # trading env arguments
    parser.add_argument('--fees', type=float, default=0.001, help='Fees for transaction costs')
    parser.add_argument('--max_ret', type=float, default=0.07, help='Maximum return in case of discrete enviroment')
    parser.add_argument('--n_ret', type=int, default=20, help='Number of returns in case of discrete enviroment')
    return parser.parse_args()
