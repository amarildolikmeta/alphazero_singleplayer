import numpy as np
import time
import multiprocessing
from tqdm import trange
import copy

USE_TQDM = True

def parallelize_eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False, max_len=np.inf):
    rewards_per_timestep = []
    lens = []
    final_states = []

    start = time.time()
    p = multiprocessing.Pool(min(n_episodes, multiprocessing.cpu_count()))
    # results = p.starmap(test, [(env) for i in range(n_episodes)])

    results = p.starmap(evaluate, [(add_terminal, copy.deepcopy(wrapper), i, interactive, max_len, verbose) for i in range(n_episodes)])
    print("Time to perform evaluation episodes:", time.time() - start, "s")

    for r in results:
        rewards_per_timestep.append(np.array(r[0]))
        lens.append(np.array(r[1]))
        final_states.append(r[2])

    # p.join()
    p.close()

    total_rewards = [sum(rew) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards)
    std = np.std(total_rewards)
    if verbose or True:
        print("Average Return = {0} +- {1}".format(avg, std))
    wrapper.reset()
    return total_rewards, rewards_per_timestep, lens, final_states


def eval_policy(wrapper, n_episodes=100, add_terminal=False, verbose=True, interactive=False, max_len=np.inf):
    rewards_per_timestep = []
    lens = []
    final_states = []
    print()
    for i in trange(n_episodes) if USE_TQDM else range(n_episodes):
        if not USE_TQDM:
            print('Evaluated ' + str(i) + ' of ' + str(n_episodes), end='\r')

        rew, t, final_state = evaluate(add_terminal, wrapper, i, interactive, max_len, verbose)
        rewards_per_timestep.append(rew)
        lens.append(t)
        final_states.append(final_state)

    total_rewards = [sum(rew) for rew in rewards_per_timestep]
    avg = np.mean(total_rewards)
    std = np.std(total_rewards)
    if verbose or True:
        print("Average Return = {0} +- {1}".format(avg, std))
    wrapper.reset()
    return total_rewards, rewards_per_timestep, lens, final_states

def evaluate(add_terminal, wrapper, i, interactive, max_len, verbose):
    start = time.time()
    s = wrapper.reset()
    # print("1")
    t = 0
    rew = []
    while t <= max_len:
        s = np.concatenate([s, [0]]) if add_terminal else s
        a = wrapper.pi_wrapper(s, max_depth=max_len-t)
        ns, r, done, inf = wrapper.step(a)
        s = ns
        if interactive:
            # print("Action=%f" % a.flatten())
            print("Reward=%f" % r)
            input()
        rew.append(r)
        t += 1
        if done:
            break
        else:
            wrapper.forward(a, s, r)

    if verbose:
        # print(acts)
        print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))

    #signature = wrapper.get_env().index_to_box(wrapper.get_env().get_signature()['state'])
    signature = None
    return rew, t, signature
