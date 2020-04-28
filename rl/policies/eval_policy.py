import numpy as np
import time


def eval_policy(env, pi, n_episodes, verbose=True, interactive=False, gamma=0.99):
    rewards = []
    disc_rewards = []
    num_stops = []
    avg_damages = []
    logs = []

    for i in range(n_episodes):

        start = time.time()
        s = env.reset()
        t = 0
        rew = 0
        disc_rew = 0
        num_pits = 0
        avg_damage = 0
        while True:
            a = pi(s)
            ns, r, done, inf = env.step(a[0])
            num_pits += (1 if a == 0 else 0)
            tire_damage = s[1]
            avg_damage += tire_damage
            s = ns
            if interactive:
                print("Action=%d" % a)

                print("Reward=%f" % r)
                input()
            rew += r
            disc_rew += gamma**t * r
            t += 1
            if done:
                avg_damage /= t
                break

        if verbose:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        rewards.append(rew)
        disc_rewards.append(disc_rew)
        num_stops.append(num_pits)
        avg_damages.append(avg_damage)
        logs.append({"reward": rew})

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))

    env.reset()

    return avg, std, logs, disc_rewards, num_stops, avg_damages
