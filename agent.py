import copy
import json
from statistics import mean

import numpy as np
import tensorflow as tf
import os
import time

from tqdm import trange

from helpers import is_atari_game, store_safely, Database
from rl.make_game import make_game
from model_tf2 import Model
from mcts import MCTS
from mcts_dpw import MCTSStochastic
from policies.eval_policy import eval_policy

import matplotlib.pyplot as plt


class EnvEvalWrapper(object):
    pass


DEBUG = False
DEBUG_TAXI = True

USE_TQDM = False

import errno
import os
from datetime import datetime


def save_parameters(params, game):
    mydir = os.path.join(
        os.getcwd(), "logs", game,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    try:
        os.makedirs(os.path.join(mydir, "plots"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    try:
        os.makedirs(os.path.join(mydir, "numpy_dumps"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    with open(os.path.join(mydir, "parameters.txt"), 'w') as d:
        d.write(json.dumps(params))

    return mydir, os.path.join(mydir, "numpy_dumps")


#### Agent ##
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units,
          stochastic=False, eval_freq=-1, eval_episodes=100, alpha=0.6, n_epochs=100, numpy_dump_dir='../', pre_process=None,
          visualize=False, game_params={}):
    visualizer = None

    parameter_list = {"game": game, "n_ep": n_ep, "n_mcts": n_mcts, "max_ep_len": max_ep_len, "lr": lr, "c": c,
                      "gamma": gamma, "data_size": data_size, "batch_size": batch_size, "temp": temp,
                      "n_hidden_layers": n_hidden_layers, "n_hidden_units": n_hidden_units,"stochastic": stochastic,
                      "eval_freq": eval_freq, "eval_episodes": eval_episodes, "alpha": alpha, "n_epochs": n_epochs,
                      "out_dir": numpy_dump_dir, "pre_process": pre_process, "visualize": visualize, "game_params": game_params}

    save_dir, numpy_dump_dir = save_parameters(parameter_list, game)

    if DEBUG_TAXI:
        from utils.visualization.taxi import TaxiVisualizer
        with open("grid2.txt", 'r') as f:
            m = f.readlines()
            matrix = []
            for r in m:
                row = []
                for ch in r.strip('\n'):
                    row.append(ch)
                matrix.append(row)
            visualizer = TaxiVisualizer(matrix)
            f.close()

    ''' Outer training loop '''
    if pre_process is not None:
        pre_process()

    # tf.reset_default_graph()

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
    mcts_params = dict(
        gamma=gamma
    )
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

    pi_loss = []
    V_loss = []
    overall_return = []

    avgs = []
    stds = []

    for ep in trange(n_ep) if USE_TQDM else range(n_ep):
        ep_pi_loss = []
        ep_V_loss = []

        if DEBUG_TAXI:
            visualizer.reset()

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
                # env_wrapper.mcts.forward(a, s, r)
                pass

            env_wrapper.reset = reset_env
            env_wrapper.step = lambda x: Env.step(x)
            env_wrapper.forward = forward
            env_wrapper.episode_probabilities = []
            env_wrapper.curr_probs = None

            def pi_wrapper(ob):
                if not is_atari:
                    mcts_env = None
                # env_wrapper.mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)
                # state, pi, V = env_wrapper.mcts.return_results(temp=0)
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

            avgs.append(np.mean(rews))
            stds.append(np.std(rews))

            # Plot the mean and variance with a whiskers plot
            plt.figure()
            plt.errorbar([10 * i for i in range(1, len(avgs) + 1)], avgs, stds, linestyle='None', marker='^', capsize=3)
            plt.xlabel("Step of evaluation")
            plt.ylabel("Return")
            plt.title("Mean and variance for return in policy evaluation")
            # plt.show()
            plt.savefig(save_dir + "/plots/meanvariance.png")
            plt.close()

        start = time.time()
        s = start_s = Env.reset()
        R = 0.0  # Total return counter
        a_store = []
        seed = np.random.randint(1e7)  # draw some Env seed
        Env.seed(seed)
        if is_atari:
            mcts_env.reset()
            mcts_env.seed(seed)

        if ep % eval_freq == 0:
            print("\nCollecting %d episodes" % eval_freq)
        mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim,
                          **mcts_params)  # the object responsible for MCTS searches

        # TODO parallelize here, very slow
        print("\nPerforming MCTS steps\n")

        prev = None
        ep_steps = 0
        rets = []
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

            # if np.array_equal(start_s, state):
            #     print("Pi target for starting state:", pi)
            D.store((state, V, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)

            s1, r, terminal, _ = Env.step(a)

            if DEBUG_TAXI:
                olds, olda = copy.deepcopy(s1), copy.deepcopy(a)
                visualizer.visualize_taxi(olds, olda)

            R += r
            if True:
                print("Reward!", r)
                rets.append(r)
            t_total += n_mcts  # total number of environment steps (counts the mcts steps)

            ep_steps = st + 1

            if terminal:
                # TODO remove this
                # if r > 0:
                #     print("Terminal state reward: ", r)
                break
            else:
                mcts.forward(a, s1, r)

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
            print(rets)
        plt.figure()
        plt.plot(online_scores)
        plt.grid = True
        plt.title("Return over policy improvement episodes")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim = 3.0
        plt.savefig(save_dir + "/plots/return.png")
        plt.close()

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R

        print()

        # Train
        # D.reshuffle()
        try:
            print("\nTraining network")
            ep_V_loss = []
            ep_pi_loss = []
            for _ in range(n_epochs):
                D.reshuffle()
                for sb, Vb, pib in D:
                    if DEBUG:
                        print("sb:", sb)
                        print("Vb:", Vb)
                        print("pib:", pib)
                    loss = model.train(sb, Vb, pib)
                    if ep % eval_freq == 0 or True:
                        ep_V_loss.append(loss[1])
                        ep_pi_loss.append(loss[2])

            # Plot the loss over training epochs

            plt.figure()
            plt.plot(ep_V_loss, label="V_loss")
            plt.plot(ep_pi_loss, label="pi_loss")
            plt.grid = True
            plt.title("Training loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(save_dir + "/plots//train_" + str(ep) + ".png")
            # plt.show()
            plt.close()

        except Exception as e:
            print("Something wrong while training")

        # model.save(out_dir + 'model')

        ep_V_loss = mean(ep_V_loss)
        ep_pi_loss = mean(ep_pi_loss)
        print()
        print("Episode", ep)
        print("pi_loss:", ep_pi_loss)
        print("V_loss:", ep_V_loss)
        V_loss.append(ep_V_loss)
        pi_loss.append(ep_pi_loss)

        # Plot the loss over different episodes

        plt.figure()
        plt.plot(V_loss, label="V_loss")
        plt.plot(pi_loss, label="pi_loss")
        plt.grid = True
        plt.title("Loss over policy improvement episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.ylim = 3.0
        plt.legend()
        plt.savefig(save_dir + "/plots/overall.png")
        # plt.show()
        plt.close()

        print("\nStart policy: ", model.predict_pi(start_s))
        print("Start value:", model.predict_V(start_s))
        print()

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best, offline_scores, V_loss, pi_loss
