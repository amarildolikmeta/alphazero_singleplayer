import argparse


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Blackjack_pi-v0', help='Training environment')
    parser.add_argument('--grid', type=str, default="grid.txt", help='TXT file specfying the game grid')
    parser.add_argument('--n_ep', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--chain_dim', type=int, default=20, help='Chain dimension')
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
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--mcts_only', action='store_true')
    parser.add_argument('--show_plots', action='store_true')
    parser.add_argument('--particles', type=int, default=0,
                        help='Numbers of particles to approximate state distributions')
    parser.add_argument('--unbiased', action='store_true', help='Use the unbiased particle algorithm')
    parser.add_argument('--variance', action='store_true', help='use variance based selection policy')
    parser.add_argument('--max_workers', type=int, default=100, help='Maximum number of parallel workers')
    parser.add_argument('--budget', type=int, default=1000, help='Computational budget')

    return parser.parse_args()