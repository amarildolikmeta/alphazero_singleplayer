from functools import partial

import numpy as np
import time
import multiprocessing
from tqdm import trange
import copy
from utils import plotter
import os
USE_TQDM = True

def finalize(rewards_per_timestep, verbose):
    total_rewards = [np.sum(rew, axis=0) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards, axis=0)
    std = np.std(total_rewards, axis=0)
    if verbose or True:
        print("Average Return = {0} +- {1}".format(avg, std))
    return total_rewards

def save_result(r, rewards_per_timestep, ep_lengths, action_counts, out_dir, results):
    r = r[0] # The result passed to the callback is a single element list
    rewards_per_timestep.append(np.array(r[0]))
    ep_lengths.append(np.array(r[1]))
    action_counts.append(r[2])
    if out_dir is not None:
        results.append(np.sum(r[0], axis=0))
    if out_dir is not None:
        np.save(out_dir + '/results.npy', results)


def parallelize_eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False,
                            max_len=200, max_workers=12, out_dir=None):
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []

    # Run the evaluation on multiple threads
    n_workers = min(n_episodes, multiprocessing.cpu_count())
    n_workers = min(n_workers, max_workers)

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    res = []
    iterations = max(n_episodes // n_workers, 1)
    remainder = n_episodes % n_workers if n_workers < n_episodes else 0

    #TODO transform in queue style, batch might be delayed even if most of the workers already finished

    # for it in range(iterations):
    #     execute_batch(action_counts, add_terminal, ep_lengths, interactive, max_len, n_workers, out_dir, res,
    #                   rewards_per_timestep, verbose, wrapper)
    # if remainder > 0:
    #     execute_batch(action_counts, add_terminal, ep_lengths, interactive, max_len, remainder, out_dir, res,
    #                   rewards_per_timestep, verbose, wrapper)

    start = time.time()
    p = multiprocessing.Pool(n_workers)

    callback_func = partial(save_result,
                            rewards_per_timestep=rewards_per_timestep,
                            ep_lengths=ep_lengths,
                            action_counts=action_counts,
                            out_dir=out_dir,
                            results=res)

    results = p.starmap_async(evaluate,
                              [(add_terminal, copy.deepcopy(wrapper), i, interactive, max_len, verbose) for i in
                               range(iterations)], callback=callback_func).get()
    print("Time to perform evaluation episodes:", time.time() - start, "s")

    # Unpack results, just for double check
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []
    for r in results:
        rewards_per_timestep.append(np.array(r[0]))
        ep_lengths.append(np.array(r[1]))
        action_counts.append(r[2])
        if out_dir is not None:
            res.append(np.sum(r[0], axis=0))
    if out_dir is not None:
        np.save(out_dir + '/results_async.npy', res)
    # p.join()
    p.close()

    total_rewards = finalize(rewards_per_timestep, verbose)

    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def execute_batch(action_counts, add_terminal, ep_lengths, interactive, max_len, n_workers, out_dir, res,
                  rewards_per_timestep, verbose, wrapper):
    start = time.time()
    p = multiprocessing.Pool(n_workers)
    results = p.starmap_async(evaluate,
                              [(add_terminal, copy.deepcopy(wrapper), i, interactive, max_len, verbose) for i in
                               range(n_workers)]).get()
    print("Time to perform evaluation episodes:", time.time() - start, "s")
    # Unpack results
    for r in results:
        rewards_per_timestep.append(np.array(r[0]))
        ep_lengths.append(np.array(r[1]))
        action_counts.append(r[2])
        if out_dir is not None:
            res.append(np.sum(r[0], axis=0))
    if out_dir is not None:
        np.save(out_dir + '/results.npy', res)
    # p.join()
    p.close()


def eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False, max_len=200,
                visualize=False, out_dir=None, render=False):
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        res = []
    for i in trange(n_episodes) if USE_TQDM else range(n_episodes):
        if not USE_TQDM:
            print('Evaluated ' + str(i) + ' of ' + str(n_episodes), end='\r')

        rew, t, count = evaluate(add_terminal, wrapper, i, interactive, max_len, verbose, visualize=visualize,
                                 render=render)
        rewards_per_timestep.append(np.array(rew))
        if out_dir is not None:
            res.append(np.sum(rew, axis=0))
        ep_lengths.append(t)
        action_counts.append(count)

    total_rewards = finalize(rewards_per_timestep, verbose)
    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def evaluate(add_terminal, wrapper, i, interactive, max_len, verbose, visualize=False, render=False):
    action_counter = 0
    n_agents = wrapper.agents_count
    start = time.time()
    s = wrapper.reset()
    t = 0
    rew = []
    inf = {}
    while t//n_agents <= max_len:
        s = np.concatenate([s, [0]]) if add_terminal else s
        a = wrapper.pi_wrapper(s, t, max_len - t)

        # Check if the action is a pit_stop
        if a == 0:
            action_counter += 1

        ns, r, done, inf = wrapper.step(a)
        if "save_path" in inf:
            save_path = inf['save_path']
            if bool(save_path):
                with open(save_path, 'a') as text_file:
                        prices = ','.join(str(e) for e in ns[:-1])
                        # toprint = prices+','+str(a-1)+',real \n'
                        toprint = prices+','+str(a-1)+','+str(r)+'\n'
                        text_file.write(toprint)

        if visualize:
            wrapper.visualize()
        if render:
            wrapper.render()

        s = ns
        if interactive:
            print("Reward=%f" % r)
            input()
        rew.append(r)
        t += 1
        if done:
            break
        else:
            wrapper.forward(a, s, r)
    if "save_path" in inf:
        plotter.data_p(inf['save_path'])

    if verbose:
        print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))

    # signature = wrapper.get_env().index_to_box(wrapper.get_env().get_signature()['state'])

    return rew, t, action_counter
