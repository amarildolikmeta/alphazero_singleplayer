import numpy as np
import time
import multiprocessing
from tqdm import trange
import copy
from utils import plotter
USE_TQDM = True


def parallelize_eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False,
                            max_len=200, max_workers=12):
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []

    # Run the evaluation on multiple threads
    start = time.time()
    n_workers = min(n_episodes, multiprocessing.cpu_count())
    n_workers = min(n_workers, max_workers)
    p = multiprocessing.Pool(n_workers)

    results = p.starmap(evaluate, [(add_terminal, copy.deepcopy(wrapper), i, interactive, max_len, verbose) for i in
                                   range(n_episodes)])
    print("Time to perform evaluation episodes:", time.time() - start, "s")

    # Unpack results
    for r in results:
        rewards_per_timestep.append(np.array(r[0]))
        ep_lengths.append(np.array(r[1]))
        action_counts.append(r[2])

    # p.join()
    p.close()

    total_rewards = [np.sum(rew) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards)
    std = np.std(total_rewards)
    if verbose or True:
        print("Average Return = {0} +- {1}".format(avg, std))
    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False, max_len=200,
                visualize=False):
    rewards_per_timestep = []
    ep_lengths = []
    action_counts = []
    print()
    for i in trange(n_episodes) if USE_TQDM else range(n_episodes):
        if not USE_TQDM:
            print('Evaluated ' + str(i) + ' of ' + str(n_episodes), end='\r')

        rew, t, count = evaluate(add_terminal, wrapper, i, interactive, max_len, verbose, visualize=visualize)
        rewards_per_timestep.append(np.array(rew))
        ep_lengths.append(t)
        action_counts.append(count)

    total_rewards = [np.sum(rew) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards)
    std = np.std(total_rewards)
    if verbose or True:
        print("Average Return = {0} +- {1}".format(avg, std))
    wrapper.reset()
    return total_rewards, rewards_per_timestep, ep_lengths, action_counts


def evaluate(add_terminal, wrapper, i, interactive, max_len, verbose, visualize=False):
    action_counter = 0
    start = time.time()
    s = wrapper.reset()
    t = 0
    rew = []
    inf = {}
    while t <= max_len:
        s = np.concatenate([s, [0]]) if add_terminal else s
        a = wrapper.pi_wrapper(s, max_depth=max_len - t)

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
