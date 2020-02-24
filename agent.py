import copy

import numpy as np
import tensorflow as tf
import os
import time

from tqdm import trange

from helpers import is_atari_game, store_safely, Database
from rl.make_game import make_game
from model import Model
from mcts import MCTS
from mcts_dpw import MCTSStochastic
from policies.eval_policy import eval_policy

class EnvEvalWrapper(object):
    pass

DEBUG = False
DEBUG_TAXI = False

#### Agent ##
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units,
          stochastic=False,  eval_freq=-1, eval_episodes=100, alpha=0.6, out_dir='../', pre_process=None,
          visualize=False, game_params={}):

    visualizer = None

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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
    model = Model(Env=Env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
    t_total = 0  # total steps
    R_best = -np.Inf


    with tf.Session() as sess:
        model.sess = sess
        sess.run(tf.global_variables_initializer())

        for ep in trange(n_ep) if DEBUG else range(n_ep):
            if DEBUG_TAXI:
                visualizer.reset()

            if eval_freq > 0  and ep % eval_freq == 0: #and ep > 0
                print('Evaluating policy for {} episodes!'.format(eval_episodes))
                seed = np.random.randint(1e7)  # draw some Env seed
                Env.seed(seed)
                s = Env.reset()

                if DEBUG_TAXI:
                    visualizer.visualize_taxi(copy.deepcopy(s), None)

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
                    #env_wrapper.mcts.forward(a, s, r)
                    pass

                env_wrapper.reset = reset_env
                env_wrapper.step = lambda x: Env.step(x)
                env_wrapper.forward = forward
                env_wrapper.episode_probabilities = []
                env_wrapper.curr_probs = None
                def pi_wrapper(ob):
                    if not is_atari:
                        mcts_env = None
                    #env_wrapper.mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)
                    #state, pi, V = env_wrapper.mcts.return_results(temp=0)
                    pi = model.predict_pi(s).flatten()
                    env_wrapper.curr_probs.append(pi)
                    a = np.argmax(pi)
                    return a

                rews, lens = eval_policy(pi_wrapper, env_wrapper, n_episodes=eval_episodes, verbose=True
                                         , max_len=max_ep_len)
                offline_scores.append([np.min(rews), np.max(rews), np.mean(rews), np.std(rews),
                                       len(rews), np.mean(lens)])
                # if len(rews) < eval_episodes or len(rews) == 0:
                #     print("WTF")
                # if np.std(rews) == 0.:
                #     print("WTF 2")
                np.save(out_dir + '/offline_scores.npy', offline_scores)
            start = time.time()
            s = Env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            if is_atari:
                mcts_env.reset()
                mcts_env.seed(seed)
            if ep % eval_freq == 0:
                print("Collecting %d episodes" % eval_freq)
            mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim, **mcts_params)  # the object responsible for MCTS searches
            for t in range(max_ep_len):
                # MCTS step
                if not is_atari:
                    mcts_env = None
                mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)  # perform a forward search
                if visualize:
                    mcts.visualize()
                state, pi, V = mcts.return_results(temp)  # extract the root output
                D.store((state, V, pi))

                # Make the true step
                a = np.random.choice(len(pi), p=pi)
                a_store.append(a)

                s1, r, terminal, _ = Env.step(a)

                if DEBUG_TAXI:
                    olds, olda = copy.deepcopy(s1), copy.deepcopy(a)
                    visualizer.visualize_taxi(olds, olda)

                R += r
                t_total += n_mcts  # total number of environment steps (counts the mcts steps)

                if terminal:
                    #TODO remove this
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
            store_safely(out_dir, 'result', {'R': episode_returns, 't': timepoints})
            np.save(out_dir + '/online_scores.npy', online_scores)
            # print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
            #                                                                          np.round((time.time() - start),
            #                                                                                   1)))


            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R

            # Train
            D.reshuffle()
            try:
                if DEBUG:
                    print("Training network")
                for epoch in range(1):
                    for sb, Vb, pib in D:
                        if DEBUG:
                            print("sb:", sb)
                            print("Vb:", Vb)
                            print("pib:", pib)
                        model.train(sb, Vb, pib)
            except Exception as e:
                print("ASD")
            model.save(out_dir + 'model')
    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best, offline_scores
