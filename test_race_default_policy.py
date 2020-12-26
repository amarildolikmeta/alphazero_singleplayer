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

    race = "Shanghai_2018"
    default_policy = False

    env = RaceEnv(horizon=100, scale_reward=False, randomize_events=False, start_lap=8)

    print(env._tyre_expected_duration)

    for i in range(1):
        rews = []

        for _ in trange(100):
            env.reset(quantile_strategies=False)
            lap = 8

            done = False
            cumulative = np.zeros(env.agents_number)
            while not done:
                agent = env.get_next_agent()
                if default_policy:
                    actions = env.get_default_strategy(agent)
                    # actions = env.get_available_actions(agent)
                    prob = PROBS[len(actions)]
                    action = np.random.choice(actions, p=prob)
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
                # if lap == 27:
                #     action = env.map_compound_to_action("A3")
                # elif lap == 53:
                #     action = env.map_compound_to_action("A5")
                # else:
                #     action = 0

                # Suzuka 2016 True

                # if lap == 12:
                #     action = env.map_compound_to_action("A1")
                # elif lap == 34:
                #     action = env.map_compound_to_action("A3")
                # else:
                #     action = 0
                # start = 20
                # stop = 22
                # if lap ==start:
                #     print(env._race_sim.get_cur_lap())
                #     env.add_fcy_custom("VSC", stop)

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

                # China 2018 true
                # if lap == 20:
                #     action = env.map_compound_to_action("A3")
                #
                # else:
                #     action = 0

                # Spa 2017 true
                # if lap == 13:
                #     action = env.map_compound_to_action("A3")
                # elif lap == 29:
                #     action = env.map_compound_to_action("A5")
                # else:
                #     action = 0

                # Sochi 2017 true
                # if lap == 34:
                #     action = env.map_compound_to_action("A4")
                #
                # else:
                #     action = 0

                # China 2018 - Planner
                if lap == 9:
                    action = env.map_compound_to_action("A3")
                elif lap == 19:
                    action = env.map_compound_to_action("A4")
                else:
                    action = 0

                s, r, done, _ = env.partial_step(action, agent)
                # TODO state compression
                # sig = env.get_signature()
                # env.set_signature(sig)

                # print(env.used_compounds)
                # print(env._pit_counts)
                cumulative += r
                lap += 1
                if done:
                    rews.append(cumulative.tolist())
            #env.save_results(race +"/" + timestamp)
        # print(rews)
        print("Return:", np.mean(rews, axis=0))
        print("std: +-", np.std(rews, axis=0))
        rews = np.array(rews)

    policy_type = "default" if default_policy else "human"
    save_location = "data/RaceStrategy-v2/{}_policy/results_unnorm_{}.npy".format(policy_type, race)
    #np.save(save_location, rews)

