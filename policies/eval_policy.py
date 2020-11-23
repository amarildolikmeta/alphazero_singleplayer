from collections import defaultdict
from functools import partial

import numpy as np
import time
import multiprocessing
from tqdm import trange, tqdm
import copy
from utils import plotter
import os

from utils.logging import Logger

USE_TQDM = True

class EvaluationResult(object):
    def __init__(self, ret, ep_length, action_counts, episode_id, execution_time):
        self.return_per_timestep = ret
        self.ep_length = ep_length
        self.action_counts = action_counts
        self.episode_id = episode_id
        self.execution_time = execution_time

def finalize(rewards_per_timestep):
    total_rewards = [np.sum(rew, axis=0) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards, axis=0)
    std = np.std(total_rewards, axis=0)
    print("Average Return = {0} +- {1}".format(avg, std))
    return total_rewards

def save_result(res:EvaluationResult, rewards_per_timestep:list, ep_lengths:list,
                action_counts:list, ep_returns:list, pbar=None) -> None:

    rewards_per_timestep.append(np.array(res.return_per_timestep))
    ep_lengths.append(np.array(res.ep_length))
    action_counts.append(res.action_counts)
    ep_returns.append(np.sum(res.return_per_timestep, axis=0))
    Logger().log_episode(res)
    Logger().save_numpy(ep_returns)

    # Update the tqdm progress bar
    if pbar:
        pbar.update(1)


def parallelize_eval_policy(wrapper, n_episodes=100, add_terminal=False, interactive=False,
                            max_len=200, max_workers=12, out_dir=None):

    if Logger().verbosity_level > 0:
        print("\n[WARNING] Verbose argument may cause inconsistent logging when performing parallel evaluation\n")

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

    start = time.time()
    p = multiprocessing.Pool(n_workers)

    for i in range(n_episodes):
        Logger().create_experiment()

    with tqdm(total=n_episodes) as pbar:

        callback_func = partial(save_result,
                                rewards_per_timestep=rewards_per_timestep,
                                ep_lengths=ep_lengths,
                                action_counts=action_counts,
                                ep_returns=res,
                                pbar=pbar)

        # Use the async style to avoid the pool waiting on possibly longer experiments,
        # each experiment is logged independently
        for i in range(n_episodes):
            index = Logger().experiments[i] if Logger().enable_neptune else i
            p.apply_async(evaluate, args=(add_terminal, copy.deepcopy(wrapper), index, interactive, max_len),
                                    callback=callback_func)
        p.close()
        p.join()

    print("Time to perform evaluation episodes:", time.time() - start, "s")

    total_rewards = finalize(rewards_per_timestep)

    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def eval_policy(wrapper, n_episodes=100, add_terminal=False, interactive=False, max_len=200,
                visualize=False, out_dir=None, render=False):
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    res = []

    for i in trange(n_episodes) if USE_TQDM else range(n_episodes):
        Logger().create_experiment()
        index = Logger().experiments[i] if Logger().enable_neptune else i
        if not USE_TQDM:
            print('Evaluated ' + str(i) + ' of ' + str(n_episodes), end='\r')

        result = evaluate(add_terminal, wrapper, index, interactive, max_len, visualize=visualize,
                                 render=render)

        save_result(result, rewards_per_timestep, ep_lengths, action_counts, res)

    total_rewards = finalize(rewards_per_timestep)
    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def evaluate(add_terminal, wrapper, episode_id, interactive, max_len, visualize=False, render=False):

    action_counter = [0] * wrapper.get_action_space()
    n_agents = wrapper.agents_count
    start = time.time()
    s = wrapper.reset()
    t = 0
    rew = []
    inf = {}

    while t//n_agents <= max_len:
        s = np.concatenate([s, [0]]) if add_terminal else s
        a = wrapper.pi_wrapper(s, t, max_len - t)

        action_counter[a] += 1
        ns, r, done, inf = wrapper.step(a)

        Logger().log_action_reward(t, a, r, episode_id)

        if "save_path" in inf:
            Logger().save_prices(inf, ns, a, r)

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

    execution_time = time.time() - start

    if Logger().verbosity_level > 0:
        print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(episode_id, rew, t, execution_time))

    # signature = wrapper.get_env().index_to_box(wrapper.get_env().get_signature()['state'])

    return EvaluationResult(rew, t, action_counter, episode_id, execution_time)
