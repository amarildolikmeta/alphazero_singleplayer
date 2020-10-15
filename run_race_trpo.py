import os
import sys
from datetime import datetime

import tensorflow.compat.v1 as tf
from tqdm import trange

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
import random
import numpy as np
from mpi4py import MPI
from rl import logger
from rl import trpo_mpi
from envs.race_strategy import Race
from rl.common.models import mlp
import utils.tf_util as U
from rl.policies.eval_policy import eval_policy
import time
import argparse
from rl.make_game import make_game
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def train_trpo(game, num_timesteps, eval_episodes, seed, horizon, out_dir='.', load_path=None, checkpoint_path_in=None,
               gamma=0.99, timesteps_per_batch=500, num_layers=0, num_hidden=32, checkpoint_freq=20, max_kl=0.01,
               ent_coef=0.):
    start_time = time.time()
    clip = None
    dir = game
    game_params = {}

    # Accept custom grid if the environment requires it
    if game == 'Taxi' or game == 'TaxiEasy':
        game_params['grid'] = args.grid
        game_params['box'] = True
    if game in ['RaceStrategy-v0', 'Cliff-v0', 'RaceStrategy-v2']:
        game_params['horizon'] = horizon

    # env = Race(gamma=gamma, horizon=horizon, )
    # env_eval = Race(gamma=gamma, horizon=horizon)
    env = make_game(args.game, game_params)
    env_eval = make_game(args.game, game_params)
    directory_output = (dir + '/trpo_' + str(num_layers) + '_' + str(num_hidden) + '_' + str(max_kl))

    def eval_policy_closure(**args):
        return eval_policy(env=env_eval, gamma=gamma, **args)

    tf.set_random_seed(seed)
    sess = U.single_threaded_session()
    sess.__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M')
    out_dir += timestamp

    time_str = timestamp
    if rank == 0:
        logger.configure(dir=out_dir + '/' + directory_output + '/logs',
                         format_strs=['stdout', 'csv'], suffix=time_str)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    network = mlp(num_hidden=num_hidden, num_layers=num_layers)

    optimized_policy = trpo_mpi.learn(network=network, env=env, eval_policy=eval_policy_closure,
                                      timesteps_per_batch=timesteps_per_batch,
                                      max_kl=max_kl, cg_iters=10, cg_damping=1e-3,
                                      total_timesteps=num_timesteps, gamma=gamma, lam=1.0, vf_iters=3, vf_stepsize=1e-4,
                                      checkpoint_freq=checkpoint_freq,
                                      checkpoint_dir_out=out_dir + '/' + directory_output + '/models/' + time_str + '/',
                                      load_path=load_path, checkpoint_path_in=checkpoint_path_in,
                                      eval_episodes=eval_episodes,
                                      init_std=1,
                                      trainable_variance=True,
                                      trainable_bias=True,
                                      clip=clip,
                                      ent_coef=ent_coef)

    s = env.reset()
    done = False

    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M')
    timestamp += "_trpo_" + str(num_layers) + '_' + str(num_hidden) + '_' + str(max_kl)

    for i in range(20):
        s = env.reset()
        print()
        print("Episode:", i + 1)
        while not done:
            a, _, _, logits = optimized_policy.step([s])
            print("Logits", logits)
            print("Action", a)
            s, _, done, _ = env.step(a)
        tires = env.used_compounds[0]
        print("Used tires:", tires)
        env.save_results(timestamp)
        done = False

    # states = []
    # actions = []
    # s = 0
    # delta_state = 0.2
    # while s < env.dim[0]:
    #     a, _, _, _ = optimized_policy.step([s])
    #     states.append(s)
    #     actions.append(a[0])
    #     s += delta_state
    # s = env.reset()
    # plt.plot(states, actions)
    # plt.show()
    print('TOTAL TIME:', time.time() - start_time)
    print("Time taken: %f seg" % ((time.time() - start_time)))
    print("Time taken: %f hours" % ((time.time() - start_time) / 3600))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='RaceStrategy-v2', help='Training environment')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument('--timesteps_per_batch', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_kl', type=float, default=0.001)
    parser.add_argument('--checkpoint_freq', type=int, default=50)
    parser.add_argument("--max_timesteps", type=int, default=402000,
                        help='Maximum number of timesteps')
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help='Episodes of evaluation')
    parser.add_argument("--seed", type=int, default=8,
                        help='Random seed')
    parser.add_argument('--horizon', type=int, help='horizon length for episode',
                        default=100)
    parser.add_argument('--dir', help='directory where to save data',
                        default='data/')
    parser.add_argument('--load_path', help='directory where to load model',
                        default='')
    parser.add_argument('--checkpoint_path_in', help='directory where to load model',
                        default='')
    parser.add_argument('--ent_coef', type=float, default=0.)

    args = parser.parse_args()

    out_dir = args.dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.load_path == "":
        args.load_path = None
    if args.checkpoint_path_in == "":
        args.checkpoint_path_in = None

    train_trpo(game=args.game,
               num_timesteps=args.max_timesteps,
               eval_episodes=args.eval_episodes,
               seed=args.seed,
               horizon=args.horizon,
               out_dir=out_dir,
               load_path=args.load_path,
               checkpoint_path_in=args.checkpoint_path_in,
               gamma=args.gamma,
               timesteps_per_batch=args.timesteps_per_batch,
               num_layers=args.num_layers,
               num_hidden=args.num_hidden,
               checkpoint_freq=args.checkpoint_freq,
               max_kl=args.max_kl,
               ent_coef=args.ent_coef)
