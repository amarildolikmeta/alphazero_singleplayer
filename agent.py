import copy
import json
from statistics import mean

import numpy as np
import time

from tqdm import trange

from helpers import is_atari_game, store_safely, Database
from rl.make_game import make_game
from model_tf2 import Model
from mcts import MCTS
from mcts_dpw import MCTSStochastic
from policies.eval_policy import eval_policy

from utils.logging.logger import Logger


class EnvEvalWrapper(object):
    pass


DEBUG = False
DEBUG_TAXI = False
MCTS_ONLY = False
USE_TQDM = False

REMOTE = False

import os

#### Agent ####
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units,
          stochastic=False, eval_freq=-1, eval_episodes=100, alpha=0.6, n_epochs=100, numpy_dump_dir='../',
          pre_process=None,
          visualize=False, game_params={}):
    visualizer = None

    parameter_list = {"game": game, "n_ep": n_ep, "n_mcts": n_mcts, "max_ep_len": max_ep_len, "lr": lr, "c": c,
                      "gamma": gamma, "data_size": data_size, "batch_size": batch_size, "temp": temp,
                      "n_hidden_layers": n_hidden_layers, "n_hidden_units": n_hidden_units, "stochastic": stochastic,
                      "eval_freq": eval_freq, "eval_episodes": eval_episodes, "alpha": alpha, "n_epochs": n_epochs,
                      "out_dir": numpy_dump_dir, "pre_process": pre_process, "visualize": visualize,
                      "game_params": game_params}

    logger = Logger(parameter_list, game, remote=REMOTE)

    if DEBUG_TAXI:
        from utils.visualization.taxi import TaxiVisualizer
        with open(game_params["grid"]) as f:
            m = f.readlines()
            matrix = []
            for r in m:
                row = []
                for ch in r.strip('\n'):
                    row.append(ch)
                matrix.append(row)
            visualizer = TaxiVisualizer(matrix)
            f.close()
            exit()

    ''' Outer training loop '''
    if pre_process is not None:
        pre_process()

    numpy_dump_dir = logger.numpy_dumps_dir

    if not os.path.exists(numpy_dump_dir):
        os.makedirs(numpy_dump_dir)

    episode_returns = []  # storage
    timepoints = []

    # Environments
    Env = make_game(game, game_params)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game, game_params) if is_atari else None
    online_scores = []
    offline_scores = []
    mcts_params = dict(gamma=gamma)

    if stochastic:
        mcts_params['alpha'] = alpha
        mcts_maker = MCTSStochastic
    else:
        mcts_maker = MCTS

    D = Database(max_size=data_size, batch_size=batch_size)
    model = Model(Env=Env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units, joint_networks=True)
    t_total = 0  # total steps
    R_best = -np.Inf

    # Variables for storing values to be plotted
    avgs = []
    stds = []

    for ep in trange(n_ep) if USE_TQDM else range(n_ep):

        if DEBUG_TAXI:
            visualizer.reset()

        ##### Policy evaluation step #####

        if eval_freq > 0 and ep % eval_freq == 0:  # and ep > 0
            print('--------------------------------\nEvaluating policy for {} episodes!\n'.format(eval_episodes))
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            s = Env.reset()

            mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim, **mcts_params)
            env_wrapper = EnvEvalWrapper()
            env_wrapper.mcts = mcts
            starting_states = []

            def reset_env():
                s = Env.reset()
                env_wrapper.mcts = mcts_maker(root_index=s, root=None, model=model,
                                              na=model.action_dim, **mcts_params)
                starting_states.append(s)
                if env_wrapper.curr_probs is not None:
                    env_wrapper.episode_probabilities.append(env_wrapper.curr_probs)
                env_wrapper.curr_probs = []
                return s

            def forward(a, s, r):
                if MCTS_ONLY:
                    env_wrapper.mcts.forward(a, s, r)

            env_wrapper.reset = reset_env
            env_wrapper.step = lambda x: Env.step(x)
            env_wrapper.forward = forward
            env_wrapper.episode_probabilities = []
            env_wrapper.curr_probs = None

            def pi_wrapper(ob):
                if not is_atari:
                    mcts_env = None

                if MCTS_ONLY:
                    env_wrapper.mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)
                    state, pi, V = env_wrapper.mcts.return_results(temp=0)
                    env_wrapper.curr_probs.append(pi)
                    a_w = np.argmax(pi)
                else:
                    pi_w = model.predict_pi(s).flatten()
                    env_wrapper.curr_probs.append(pi_w)
                    a_w = np.argmax(pi_w)
                return a_w

            rews, lens = eval_policy(pi_wrapper, env_wrapper, n_episodes=eval_episodes, verbose=False
                                     , max_len=max_ep_len)
            offline_scores.append([np.min(rews), np.max(rews), np.mean(rews), np.std(rews),
                                   len(rews), np.mean(lens)])
            # if len(rews) < eval_episodes or len(rews) == 0:
            #     print("WTF")
            # if np.std(rews) == 0.:
            #     print("WTF 2")
            np.save(numpy_dump_dir + '/offline_scores.npy', offline_scores)

            # Store and plot data

            avgs.append(np.mean(rews))
            stds.append(np.std(rews))

            logger.plot_evaluation_mean_and_variance(avgs, stds)

        start = time.time()
        s = start_s = Env.reset()
        R = 0.0  # Total return counter
        a_store = []
        seed = np.random.randint(1e7)  # draw some Env seed
        Env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        if eval_freq > 0 and ep % eval_freq == 0:
            print("\nCollecting %d episodes" % eval_freq)
        mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim,
                          **mcts_params)  # the object responsible for MCTS searches

        # TODO parallelize here, very slow
        print("\nPerforming MCTS steps\n")

        ep_steps = 0
        start_targets = []

        ##### Policy improvement step #####

        for st in trange(max_ep_len) if USE_TQDM else range(max_ep_len):

            if not USE_TQDM:
                print('Step ' + str(st + 1) + ' of ' + str(max_ep_len), end='\r')

            # MCTS step
            if not is_atari:
                mcts_env = None
            mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)  # perform a forward search

            if visualize:
                mcts.visualize()

            state, pi, V = mcts.return_results(temp)  # extract the root output

            # Save targets for starting state to debug
            if np.array_equal(start_s, state):
                if DEBUG:
                    print("Pi target for starting state:", pi)
                start_targets.append((V, pi))
            D.store((state, V, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)

            s1, r, terminal, _ = Env.step(a)

            # Perform command line visualization if necessary
            if DEBUG_TAXI:
                olds, olda = copy.deepcopy(s1), copy.deepcopy(a)
                visualizer.visualize_taxi(olds, olda)
                print("Reward:", r)

            R += r
            t_total += n_mcts  # total number of environment steps (counts the mcts steps)
            ep_steps = st + 1

            if terminal:
                break  # Stop the episode if we encounter a terminal state
            else:
                mcts.forward(a, s1, r)  # Otherwise proceed

        # Finished episode
        if DEBUG:
            print("Train episode return:", R)
            print("Train episode actions:", a_store)
        episode_returns.append(R)  # store the total episode return
        online_scores.append(R)
        timepoints.append(t_total)  # store the timestep count of the episode return
        store_safely(numpy_dump_dir, '/result', {'R': episode_returns, 't': timepoints})
        np.save(numpy_dump_dir + '/online_scores.npy', online_scores)

        if DEBUG or True:
            print('Finished episode {} in {} steps, total return: {}, total time: {} sec'.format(ep, ep_steps,
                                                                                                 np.round(R, 2),
                                                                                                 np.round((
                                                                                                                      time.time() - start),
                                                                                                          1)))
        # Plot the online return over training episodes

        logger.plot_online_return(online_scores)

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R

        print()

        # Train
        try:
            print("\nTraining network")
            ep_V_loss = []
            ep_pi_loss = []

            for _ in range(n_epochs):
                # Reshuffle the dataset at each epoch
                D.reshuffle()

                batch_V_loss = []
                batch_pi_loss = []

                # Batch training
                for sb, Vb, pib in D:

                    if DEBUG:
                        print("sb:", sb)
                        print("Vb:", Vb)
                        print("pib:", pib)

                    loss = model.train(sb, Vb, pib)

                    batch_V_loss.append(loss[1])
                    batch_pi_loss.append(loss[2])

                ep_V_loss.append(mean(batch_V_loss))
                ep_pi_loss.append(mean(batch_pi_loss))

            # Plot the loss over training epochs

            logger.plot_loss(ep, ep_V_loss, ep_pi_loss)

        except Exception as e:
            print("Something wrong while training:", e)

        # model.save(out_dir + 'model')

        # Plot the loss over different episodes
        logger.plot_training_loss_over_time()

        pi_start = model.predict_pi(start_s)
        V_start = model.predict_V(start_s)

        print("\nStart policy: ", pi_start)
        print("Start value:", V_start)

        logger.log_start(ep, pi_start, V_start, start_targets)

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best, offline_scores
