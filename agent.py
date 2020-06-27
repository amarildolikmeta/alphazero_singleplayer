import copy
from statistics import mean
import numpy as np
import time
from helpers import is_atari_game, store_safely, Database
from rl.make_game import make_game
from models.model_tf2 import ModelWrapper
from policies.eval_policy import eval_policy, parallelize_eval_policy
from utils.logging.logger import Logger
from utils.env_wrapper import Wrapper
from particle_filtering.parallel_sampler import ParallelSampler
import os
DEBUG = False
DEBUG_TAXI = False
USE_TQDM = True


#### Agent ####
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units,
          stochastic=False, eval_freq=-1, eval_episodes=100, alpha=0.6, n_epochs=100, c_dpw=1, numpy_dump_dir='../',
          pre_process=None, visualize=False, game_params={}, parallelize_evaluation=False, mcts_only=False,
          particles=0, show_plots=False, n_workers=1, use_sampler=False, budget=np.inf):
    visualizer = None

    # if particles:
    #     parallelize_evaluation = False  # Cannot run parallelized evaluation with particle filtering

    if not mcts_only:
        from mcts import MCTS
        from mcts_dpw import MCTSStochastic
    elif particles:
        from particle_filtering.pf_mcts_edo import PFMCTS
    else:
        from pure_mcts.mcts import MCTS
        from pure_mcts.mcts_dpw import MCTSStochastic

    if parallelize_evaluation:
        print("The evaluation will be parallel")

    parameter_list = {"game": game, "n_ep": n_ep, "n_mcts": n_mcts, "max_ep_len": max_ep_len, "lr": lr, "c": c,
                      "gamma": gamma, "data_size": data_size, "batch_size": batch_size, "temp": temp,
                      "n_hidden_layers": n_hidden_layers, "n_hidden_units": n_hidden_units, "stochastic": stochastic,
                      "eval_freq": eval_freq, "eval_episodes": eval_episodes, "alpha": alpha, "n_epochs": n_epochs,
                      "out_dir": numpy_dump_dir, "pre_process": pre_process, "visualize": visualize,
                      "game_params": game_params, "n_workers": n_workers, "use_sampler": use_sampler}

    logger = Logger(parameter_list, game, show=show_plots)

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
    game_params = {'save_dir':logger.save_dir}
    Env = make_game(game, game_params)
    num_actions = Env.action_space.n
    sampler = None
    if use_sampler:
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
    mcts_params = dict(gamma=gamma)
    if particles:
        mcts_params['particles'] = particles
        mcts_params['sampler'] = sampler
    if stochastic:
        mcts_params['alpha'] = alpha
        mcts_maker = MCTSStochastic
    elif particles:
        mcts_maker = PFMCTS
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

    model_wrapper = ModelWrapper(**model_params)

    t_total = 0  # total steps
    R_best = -np.Inf
    a_best = None
    seed_best = None

    # Variables for storing values to be plotted
    avgs = []
    stds = []

    # Run the episodes
    for ep in range(n_ep):

        if DEBUG_TAXI:
            visualizer.reset()

        ##### Policy evaluation step #####
        if eval_freq > 0 and ep % eval_freq == 0:  # and ep > 0
            print('--------------------------------\nEvaluating policy for {} episodes!'.format(eval_episodes))
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            s = Env.reset()

            if parallelize_evaluation:
                penv = None
                pgame = {"game_maker": make_game,
                         "game": game,
                         "game_params": game_params}
            else:
                penv = Env
                pgame = None

            model_file = os.path.join(logger.save_dir, "model.h5")

            # model_wrapper.save(model_file)

            env_wrapper = Wrapper(s, mcts_maker, model_file, model_params, mcts_params, is_atari, n_mcts, budget,
                                  mcts_env, c_dpw, temp, Env=penv, game_maker=pgame, mcts_only=mcts_only)

            # Run the evaluation
            if parallelize_evaluation:
                total_reward, reward_per_timestep, lens, action_counts = \
                    parallelize_eval_policy(env_wrapper, n_episodes=eval_episodes, verbose=False, max_len=max_ep_len)
            else:
                total_reward, reward_per_timestep, lens, action_counts = \
                    eval_policy(env_wrapper, n_episodes=eval_episodes, verbose=False, max_len=max_ep_len)

            # offline_scores.append([np.min(rews), np.max(rews), np.mean(rews), np.std(rews),
            #                        len(rews), np.mean(lens)])

            offline_scores.append([total_reward, reward_per_timestep, lens, action_counts])

            np.save(numpy_dump_dir + '/offline_scores.npy', offline_scores)

            # Store and plot data
            avgs.append(np.mean(total_reward))
            stds.append(np.std(total_reward))

            logger.plot_evaluation_mean_and_variance(avgs, stds)

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

            if eval_freq > 0 and ep % eval_freq == 0:
                print("\nCollecting %d episodes" % eval_freq)
            mcts = mcts_maker(root_index=s, root=None, model=model_wrapper, na=model_wrapper.action_dim,
                              **mcts_params)  # the object responsible for MCTS searches

            print("\nPerforming MCTS steps\n")

            ep_steps = 0
            start_targets = []

            for st in range(max_ep_len):

                print_step = max_ep_len // 10
                if st % print_step == 0:
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

            # Train only if the model has to be used
            if not mcts_only:
                # Train
                try:
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

                    logger.plot_loss(ep, ep_V_loss, ep_pi_loss)

                except Exception as e:
                    print("Something wrong while training:", e)

                # model.save(out_dir + 'model')

                # Plot the loss over different episodes
                logger.plot_training_loss_over_time()

                pi_start = model_wrapper.predict_pi(start_s)
                V_start = model_wrapper.predict_V(start_s)

                print("\nStart policy: ", pi_start)
                print("Start value:", V_start)

                logger.log_start(ep, pi_start, V_start, start_targets)

    # Return results
    if use_sampler:
        sampler.close()
    return episode_returns, timepoints, a_best, seed_best, R_best, offline_scores
