from datetime import datetime

from envs.race_strategy_event import RaceEnv
import numpy as np

from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

MAX_P = [1]
PROB_1 = [0.95, 0.05]
PROB_2 = [0.95, 0.025, 0.025]
PROB_3 = [0.91, 0.03, 0.03, 0.03]
PROBS = {1: MAX_P, 2: PROB_1, 3: PROB_2, 4: PROB_3}

if __name__ == '__main2__':
    env = RaceEnv(horizon=100)
    env.reset()
    env.step(1)
    print(env.get_available_actions(env.get_next_agent()))

if __name__ == '__main__':

    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M')

    env = RaceEnv(horizon=100)
    rews = []
    for i in trange(20):
        env.reset()
        lap = 1

        done = False
        cumulative = np.zeros(env.agents_number)
        while not done:
            agent = env.get_next_agent()
            # actions = env.get_available_actions(agent)
            # prob = PROBS[len(actions)]
            # action = np.random.choice(actions, p=prob)
            # if lap == 14:
            #     action = 2
            # elif lap == 37:
            #     action = 1
            # else:
            #     action = 0
            action = 0
            s, r, done, _ = env.partial_step(action, agent)
            cumulative += r
            lap += 1
            if done:
                rews.append(cumulative.tolist())
        env.save_results(timestamp + "_fixed20_lin")
    # print(rews)
    # print("Return:", np.mean(rews, axis=0))
    # print("std: +-", np.std(rews, axis=0))
    # rews = np.array(rews)
    #
    # rews_budget = np.load('./logs/RaceStrategy-v2/multiagent/2020-10-11_11-35/results.npy')
    # print(rews_budget)
    #
    # fig = plt.figure()
    # plt.title('Return distribution - Catalunya 2017')
    # sns.distplot(rews[:, 0], label='HAM def', hist=False)
    # sns.distplot(rews[:, 1], label='VET def', hist=False)
    # sns.distplot(rews_budget[:, 0], label='HAM - OL 1000b', hist=False)
    # sns.distplot(rews_budget[:, 1], label='VET - PF 1000b', hist=False)
    # plt.savefig('./logs/RaceStrategy-v2/multiagent/return_comparison_1000b_1000r.png')
    #
    #

