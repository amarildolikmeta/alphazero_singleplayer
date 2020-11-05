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

if __name__ == '__main__':

    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M')

    env = RaceEnv(horizon=100, scale_reward=False, randomize_events=True)
    rews = []
    for i in trange(1):
        env.reset()
        lap = 1

        done = False
        cumulative = np.zeros(env.agents_number)
        while not done:
            agent = env.get_next_agent()
            actions = env.get_available_actions(agent)
            prob = PROBS[len(actions)]
            action = np.random.choice(actions, p=prob)

            # Catalunya 2017 true

            # if lap == 14:
            #     action = 2
            # elif lap == 37:
            #     action = 1
            # else:
            #     action = 0

            # Australia 2017 true
            # if lap == 23:
            #     action = env.map_compound_to_action("A3")
            # else:
            #     action = 0

            s, r, done, _ = env.partial_step(action, agent)
            env.reset_stochasticity()
            # print(env.used_compounds)
            # print(env._pit_counts)
            cumulative += r
            lap += 1
            if done:
                rews.append(cumulative.tolist())
        env.save_results(timestamp + "_aus_test")
    # print(rews)
    print("Return:", np.mean(rews, axis=0))
    print("std: +-", np.std(rews, axis=0))
    rews = np.array(rews)
    #np.save("data/RaceStrategy-v2/default_policy/results_unnorm_aus.npy", rews)

