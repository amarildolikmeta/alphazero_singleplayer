import copy
from statistics import mean
import numpy as np
import time

from typing import List

from helpers import is_atari_game, Database
from race_components.helpers import load_race_agents_config
from rl.make_game import make_game
from policies.eval_policy import eval_policy, parallelize_eval_policy
from utils.env_wrapper import Wrapper
from utils.logging import Logger
from particle_filtering.parallel_sampler import ParallelSampler
import os

from utils.offline_score import OfflineScore
from utils.race_wrapper import RaceWrapper

DEBUG = False
DEBUG_TAXI = False
USE_TQDM = True

#### Agent ####
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units,
          stochastic=False, eval_freq=-1, eval_episodes=100, alpha=0.6, n_epochs=100, c_dpw=1, numpy_dump_dir='../',
          pre_process=None, visualize=False, game_params={}, parallelize_evaluation=False, mcts_only=False,
          particles=0, show_plots=False, n_workers=1, use_sampler=False, budget=np.inf, unbiased=False, biased=False,
          max_workers=100, variance=False, depth_based_bias=False, scheduler_params=None, out_dir=None,
          render=False, second_version=False, third_version=False, multiagent=False, csi=1., bayesian=False,
          q_learning=False, ucth=False, log_timestamp=None, verbose=False, power=False, p=100) -> List[OfflineScore]:
    parameter_dict = locals()  # Save the state of all variables for logging
    logger = Logger()
    logger.set_timestamp()
    logger.set_verbosity_level(int(verbose))
    logger.create_directories(game, out_dir)
    logger.save_parameters(parameter_dict)

    visualizer = None

    if not mcts_only:
        from mcts import MCTS
        from mcts_dpw import MCTSStochastic
    elif particles:
        if unbiased:
            if bayesian:
                from particle_filtering.bayesian_ol_uct import Bayesian_OL_MCTS
            elif q_learning:
                from planners.q_learning_ol_uct import QL_OL_MCTS
            elif ucth:
                from planners.q_learning_ol_uct_h import QL_UCTH_OL_MCTS
            elif power:
                from particle_filtering.power_ol_uct import PowerOLMCTS
            else:
                from particle_filtering.ol_uct import OL_MCTS
        elif biased:
            if second_version:
                from particle_filtering.pf_uct_2 import PFMCTS2 as PFMCTS
            elif third_version:
                from particle_filtering.pf_uct_3 import PFMCTS3 as PFMCTS
            else:
                from particle_filtering.pf_uct import PFMCTS
        else:
            from particle_filtering.pf_mcts_edo import PFMCTS
    else:
        from pure_mcts.mcts import MCTS
        from pure_mcts.mcts_dpw import MCTSStochastic

    if parallelize_evaluation and verbose:
        print("The evaluation will be parallel")

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

    episode_returns = []  # storage
    timepoints = []

    # Environments
    if game == 'Trading-v0':
        game_params['save_dir'] = out_dir #logger.save_dir
    if game == 'RaceStrategy-v2':
        game_params['log_path'] = Logger().save_dir

    Env = make_game(game, game_params)
    num_actions = Env.action_space.n
    sampler = None
    if use_sampler and not (unbiased or biased):
        def make_pi(action_space):
            def pi(s):
                return np.random.randint(low=0, high=action_space.n)

            return pi

        def make_env():
            return make_game(game, game_params)

        sampler = ParallelSampler(make_pi=make_pi, make_env=make_env, n_particles=particles,
                                  n_workers=n_workers, seed=10)

    is_atari = is_atari_game(Env)
    mcts_env = make_game(game, game_params) if is_atari else None
    online_scores = []
    offline_scores = []

    # Setup the parameters for generating the search environments

    if game == "RaceStrategy-v1" or game == "RaceStrategy-v2" and multiagent:
        # TODO add parameter for config file or streamline the loading
        config_file = 'envs/configs/race_strategy_full_dpw.json'
        print("\nUsing race_components MCTS")
        print("Config file: {}\n".format(config_file))
        mcts_maker, mcts_params, c_dpw = load_race_agents_config(config_file, gamma)

    else:
        mcts_params = dict(gamma=gamma)
        if particles:
            if not (biased or unbiased):
                mcts_params['particles'] = particles
                mcts_params['sampler'] = sampler
            elif biased:
                if verbose:
                    print("\nUsing PFMCTS\n")
                mcts_params['alpha'] = alpha
                mcts_maker = PFMCTS

            mcts_params['depth_based_bias'] = depth_based_bias
            if unbiased:
                if bayesian:
                    if verbose:
                        print("\nUsing Bayesian OLMCTS\n")
                    mcts_maker = Bayesian_OL_MCTS
                elif q_learning:
                    if verbose:
                        print("\nUsing Q-Learning MCTS\n")
                    mcts_maker = QL_OL_MCTS
                    mcts_params['alpha'] = alpha
                    mcts_params['beta'] = beta
                elif ucth:
                    if verbose:
                        print("\nUsing Q-Learning UCT-H\n")
                    mcts_maker = QL_UCTH_OL_MCTS
                    mcts_params['alpha'] = alpha
                elif power:
                    if verbose:
                        print("\nUsing Power OLMCTS\n")
                    mcts_maker = PowerOLMCTS
                    mcts_params['p'] = p
                else:
                    if verbose:
                        print("\nUsing OLMCTS\n")
                    mcts_params['variance'] = variance
                    mcts_params['csi'] = csi
                    mcts_maker = OL_MCTS

        elif stochastic:
            mcts_params['alpha'] = alpha
            mcts_params['depth_based_bias'] = depth_based_bias
            mcts_params['c_dpw'] = c_dpw
            mcts_maker = MCTSStochastic
        else:
            mcts_maker = MCTS

    # Prepare the database for storing training data to be sampled
    db = Database(max_size=data_size, batch_size=batch_size)

    # TODO extract dimensions to avoid allocating model
    # Setup the model
    model_params = {"Env": Env,
                    "lr": lr,
                    "n_hidden_layers": n_hidden_layers,
                    "n_hidden_units": n_hidden_units,
                    "joint_networks": True}



    t_total = 0  # total steps
    R_best = -np.Inf
    a_best = None
    seed_best = None

    # Variables for storing values to be plotted
    avgs = []
    stds = []

    # Run the episodes
    if not mcts_only:
        from models.model_tf2 import ModelWrapper
        model_wrapper = ModelWrapper(**model_params)
    else:
        model_wrapper = None
    for ep in range(n_ep):

        if DEBUG_TAXI:
            visualizer.reset()

        ##### Policy evaluation step #####
        if eval_freq > 0 and ep % eval_freq == 0:  # and ep > 0
            print('--------------------------------\nEvaluating policy for {} episodes!'.format(eval_episodes))
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            # s = Env.reset()

            if parallelize_evaluation:
                penv = None
                pgame = {"game_maker": make_game,
                         "game": game,
                         "game_params": game_params}
            else:
                penv = Env
                pgame = None

            model_file = os.path.join(out_dir, "model.h5")
            if not mcts_only:
                model_wrapper.save(model_file)

            if game == "RaceStrategy-v1" or game == "RaceStrategy-v2" and multiagent:
                raise NotImplementedError
                # env_wrapper = RaceWrapper(mcts_maker, model_file, model_params, mcts_params, is_atari, n_mcts, budget,
                #                   mcts_env, c, temp, env=penv, game_maker=pgame, mcts_only=mcts_only,
                #                   scheduler_params=scheduler_params, verbose=verbose)
            else:
                env_wrapper = Wrapper(mcts_maker, model_file, model_params, mcts_params, is_atari, n_mcts, budget,
                                      mcts_env, c, temp, env=penv, game_maker=pgame, mcts_only=mcts_only,
                                      scheduler_params=scheduler_params)

            # Run the evaluation
            if parallelize_evaluation:
                total_reward, reward_per_timestep, lens, action_counts = \
                    parallelize_eval_policy(env_wrapper, n_episodes=eval_episodes, max_len=max_ep_len,
                                            max_workers=max_workers, out_dir=out_dir)
            else:
                total_reward, reward_per_timestep, lens, action_counts = \
                    eval_policy(env_wrapper, n_episodes=eval_episodes, max_len=max_ep_len,
                                visualize=visualize, out_dir=out_dir, render=render)

            # offline_scores.append([np.min(rews), np.max(rews), np.mean(rews), np.std(rews),
            #                        len(rews), np.mean(lens)])

            offline_scores.append(OfflineScore(total_reward, reward_per_timestep, lens, action_counts))

            #np.save(numpy_dump_dir + '/offline_scores.npy', offline_scores)

            # Store and plot data
            avgs.append(np.mean(total_reward))
            stds.append(np.std(total_reward))

            #logger.plot_evaluation_mean_and_variance(avgs, stds)

        ##### Policy improvement step #####

        if not mcts_only:

            start = time.time()
            s = start_s = Env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            if is_atari:
                mcts_env.reset()
                mcts_env.seed(seed)

            if eval_freq > 0 and ep % eval_freq == 0 and verbose:
                print("\nCollecting %d episodes" % eval_freq)
            mcts = mcts_maker(root_index=s, root=None, model=model_wrapper, na=model_wrapper.action_dim,
                              **mcts_params)  # the object responsible for MCTS searches

            if verbose:
                print("\nPerforming MCTS steps\n")

            ep_steps = 0
            start_targets = []

            for st in range(max_ep_len):

                print_step = max_ep_len // 10
                if st % print_step == 0 and verbose:
                    print('Step ' + str(st + 1) + ' of ' + str(max_ep_len))

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
                db.store((state, V, pi))

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
            #store_safely(numpy_dump_dir, '/result', {'R': episode_returns, 't': timepoints})
            #np.save(numpy_dump_dir + '/online_scores.npy', online_scores)

            if DEBUG or verbose:
                print('Finished episode {} in {} steps, total return: {}, total time: {} sec'.format(ep, ep_steps,
                                                                                                     np.round(R, 2),
                                                                                                     np.round((
                                                                                                             time.time() - start),
                                                                                                         1)))
            # Plot the online return over training episodes

            #logger.plot_online_return(online_scores)

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R

            print()

            # Train only if the model has to be used
            if not mcts_only:
                # Train
                try:
                    if verbose:
                        print("\nTraining network")
                    ep_V_loss = []
                    ep_pi_loss = []

                    for _ in range(n_epochs):
                        # Reshuffle the dataset at each epoch
                        db.reshuffle()

                        batch_V_loss = []
                        batch_pi_loss = []

                        # Batch training
                        for sb, Vb, pib in db:

                            if DEBUG:
                                print("sb:", sb)
                                print("Vb:", Vb)
                                print("pib:", pib)

                            loss = model_wrapper.train(sb, Vb, pib)

                            batch_V_loss.append(loss[1])
                            batch_pi_loss.append(loss[2])

                        ep_V_loss.append(mean(batch_V_loss))
                        ep_pi_loss.append(mean(batch_pi_loss))

                    # Plot the loss over training epochs

                    #logger.plot_loss(ep, ep_V_loss, ep_pi_loss)

                except Exception as e:
                    print("Something wrong while training:", e)

                # model.save(out_dir + 'model')

                # Plot the loss over different episodes
                #logger.plot_training_loss_over_time()

                pi_start = model_wrapper.predict_pi(start_s)
                V_start = model_wrapper.predict_V(start_s)
                if verbose:
                    print("\nStart policy: ", pi_start)
                    print("Start value:", V_start)

                #logger.log_start(ep, pi_start, V_start, start_targets)

    # Return results
    if use_sampler:
        sampler.close()
    return episode_returns, timepoints, a_best, seed_best, R_best, offline_scores
