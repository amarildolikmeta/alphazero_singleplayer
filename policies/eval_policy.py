import numpy as np
import time


def eval_policy(pi,  env, n_episodes=100, add_terminal=False, verbose=True, interactive=False, max_len=np.inf):

    rewards = []
    lens = []
    for i in range(n_episodes):
        start = time.time()
        s = env.reset()
        t = 0
        rew = 0
        while t <= max_len:
            s = np.concatenate([s, [0]]) if add_terminal else s
            a = pi(s)
            ns, r, done, inf = env.step(a)
            s = ns
            if interactive:
                #print("Action=%f" % a.flatten())
                print("Reward=%f" % r)
                input()
            rew += r
            t += 1
            if done:
                break
            else:
                env.forward(a, s, r)

        if verbose and False:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        rewards.append(rew)
        lens.append(t)

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))
    env.reset()
    return rewards, lens
