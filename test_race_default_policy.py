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

    env = RaceEnv(horizon=100, scale_reward=False, randomize_events=False, start_lap=8)

    # print("Default Strategies")
    # strategy = [[0, "A5", 2, 0.0], [22, "A3", 0, 0.0], [42, "A3", 0, 0.0]]
    # env.reset()
    # print(env.simulate_strategy(env._pars_in, 'VET', strategy)[40: 45])
    # strategy = [[0, "A5", 2, 0.0], [22, "A3", 0, 0.0], [42, "A4", 0, 0.0]]
    # env.reset()
    # print(env.simulate_strategy(env._pars_in, 'VET', strategy)[40: 45])
    # strategy = [[0, "A5", 2, 0.0], [22, "A3", 0, 0.0]]
    env.reset()
    print(env._tyre_expected_duration)
    # print(env.simulate_strategy(env._pars_in, 'VET', strategy)[40: 45])
    rews = []

    #print(env.map_compound_to_action("A3"))

    #for special_action in [env.map_compound_to_action("A3"), env.map_compound_to_action("A4"), 0]:
    for i in trange(100):
        env.reset()
        lap = 8

        done = False
        cumulative = np.zeros(env.agents_number)
        while not done:
            agent = env.get_next_agent()
            # actions = env.get_default_strategy(agent)
            # # actions = env.get_available_actions(agent)
            # prob = PROBS[len(actions)]
            # action = np.random.choice(actions, p=prob)
            # if action > 0:
            #     print(lap, env.map_action_to_compound(action))

            # Catalunya 2017 true

            # if lap == 14:
            #     action = 2
            # elif lap == 37:
            #     action = 1
            # else:
            #     action = 0

            #Australia 2017 true
            # if lap == 23:
            #     action = env.map_compound_to_action("A3")
            # else:
            #     action = 0

            # Brazil 2018 true
            if lap == 27:
                action = env.map_compound_to_action("A3")
            elif lap == 53:
                action = env.map_compound_to_action("A5")
            else:
                action = 0

            # Suzuka 2016 True

            # if lap == 12:
            #     action = env.map_compound_to_action("A1")
            # elif lap == 34:
            #     action = env.map_compound_to_action("A3")
            # else:
            #     action = 0

            # Suzuka 2015 true
            # if lap == 13:
            #     action = env.map_compound_to_action("A1")
            # elif lap == 30:
            #     action = env.map_compound_to_action("A1")
            # else:
            #     action = 0


            # Austria 2017 true
            # if lap == 34:
            #     action = env.map_compound_to_action("A4")

            # else:
            #     action = 0



            s, r, done, _ = env.partial_step(action, agent)
            # TODO state compression
            sig = env.get_signature()
            env.set_signature(sig)

            # TODO safety car randomization (enable randomize_events in constructor)
            # env.reset_stochasticity()
            # print(env.used_compounds)
            # print(env._pit_counts)
            cumulative += r
            lap += 1
            if done:
                rews.append(cumulative.tolist())
        env.save_results("/bra_2018/" + timestamp)
    # print(rews)
    print("Return:", np.mean(rews, axis=0))
    print("std: +-", np.std(rews, axis=0))
    rews = np.array(rews)
    #np.save("data/RaceStrategy-v2/default_policy/results_unnorm_aus.npy", rews)

