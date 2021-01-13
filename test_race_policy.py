from collections import defaultdict
from datetime import datetime

from tqdm import trange

from envs.race_strategy_event import RaceEnv, MCS_PARS_FILE, PROBS
from race_simulation.racesim.src.import_pars import import_pars

import numpy as np


def simulate_strategy(race, driver, strategy_type):
    today = datetime.now()
    timestamp = today.strftime('%Y-%m-%d_%H-%M')

    true_policy = False
    quantile_strategies = False
    default_policy = False

    if strategy_type == "true":
        true_policy = True
    elif strategy_type == "quantile":
        quantile_strategies = True
    else:
        default_policy = True

    env = RaceEnv(horizon=100, scale_reward=False, start_lap=8,
                  config_path='./envs/configs/race_strategy_event_env_config.json')
    env.race_config = "pars_" + race + ".ini"
    env.reset()
    env.enable_search_mode()

    csv_save_prefix = race + '/' + strategy_type + '/' + timestamp

    if true_policy:
        pars_in, vse_paths = import_pars(use_print=False,
                                         use_vse=False,
                                         race_pars_file=env.race_config,
                                         mcs_pars_file=MCS_PARS_FILE)

        strategy = defaultdict(int)

        true_strategy = pars_in['driver_pars'][driver]['strategy_info']

        if len(true_strategy) == 1:
            print("[WARNING] No pit stop in true strategy!")
        else:
            for pit in true_strategy[1:]:
                strategy[pit[0]] = env.map_compound_to_action(pit[1])

    for i in range(1):
        rews = []

        for _ in trange(100):
            env.reset(quantile_strategies=quantile_strategies)
            env.enable_search_mode()
            lap = 8

            done = False
            cumulative = np.zeros(env.agents_number)
            while not done:
                agent = env.get_next_agent()
                if default_policy or quantile_strategies:
                    actions = env.get_default_strategy(agent)
                    # actions = env.get_available_actions(agent)
                    prob = PROBS[len(actions)]
                    action = np.random.choice(actions, p=prob)
                    # if action > 0:
                    #     print(lap, env.map_action_to_compound(action))
                elif true_policy:
                    action = strategy[lap]

                s, r, done, _ = env.partial_step(action, agent)

                cumulative += r
                lap += 1
                if done:
                    rews.append(cumulative.tolist())

            env.save_results(csv_save_prefix)
        # print(rews)
        print("Return:", np.mean(rews, axis=0))
        print("std: +-", np.std(rews, axis=0))
        rews = np.array(rews)

    policy_type = "default" if default_policy or quantile_strategies else "human"
    numpy_save_location = "data/RaceStrategy-v2/{}_policy/results_unnorm_{}.npy".format(policy_type, race)
    np.save(numpy_save_location, rews)
    return numpy_save_location, csv_save_prefix

if __name__ == '__main__':
    simulate_strategy("SaoPaulo_2018", "VET", "default")