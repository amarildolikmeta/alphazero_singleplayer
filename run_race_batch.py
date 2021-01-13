import os
import json

if __name__ == '__main__':
    # Load configuration information from the json file
    with open('launch_scripts/configs/race_mcts_planner_config.json') as config_file:
        config_data = json.load(config_file)
        config_file.close()

    budget = config_data['budget']
    parallel = '--parallel' if config_data['parallel_execution'] else ''
    max_workers = config_data['max_workers']
    c = config_data['c']
    enable_scheduler = '--budget_scheduler' if config_data['enable_scheduler'] else ''
    scheduler_params = config_data['scheduler_params']
    races = config_data['races']['race_list']
    race_parameters = config_data['races']['race_config']
    eval_episodes = config_data['eval_episodes']

    # Build command string
    #command = 'taskset -c 1-22 python3 alphazero.py --game=RaceStrategy-v2 --budget={} --gamma=1 --max_ep_len=100 ' \
    command = 'python3 alphazero.py --game=RaceStrategy-v2 --budget={} --gamma=1 --max_ep_len=100 ' \
              '--eval_freq=1 --temp=0 --n_ep=1 --eval_episodes={} --mcts_only --particles=1 ' \
              '--n_experiments=1 --unbiased {} --max_workers={} {} --min_depth={} --slope={} --min_budget={} ' \
              '--q_learning --c={}'.format(budget, eval_episodes, parallel, max_workers, enable_scheduler,
                                            scheduler_params['min_depth'],
                                            scheduler_params['slope'],
                                            scheduler_params['min_budget'], c)

    # Backup old config file
    os.rename(r'envs/configs/race_strategy_event_env_config.json',
              r'envs/configs/race_strategy_event_env_config_backup.json')

    # Execute the experiment for each desired race
    for race in races:
        print('### Race:', race['track'], str(race['year']), '###')
        print()
        race_parameters['track'] = race['track']
        race_parameters['year'] = race['year']

        # Create the config file for the current race
        with open('envs/configs/race_strategy_event_env_config.json', 'w') as config_file:
            json.dump(race_parameters, config_file)
            config_file.close()

        # Execute
        os.system(command)
        # print(command)
        print()
        print()

    # Remove config file and restore backup
    os.remove(r'envs/configs/race_strategy_event_env_config.json')
    os.rename(r'envs/configs/race_strategy_event_env_config_backup.json',
              r'envs/configs/race_strategy_event_env_config.json')