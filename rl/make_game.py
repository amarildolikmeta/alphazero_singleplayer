# -*- coding: utf-8 -*-
"""
Custom game generation function
@author: thomas
"""
import gym
import numpy as np
from gym.envs.registration import register

from envs import generate_taxi, generate_taxi_easy, generate_arms, generate_river, generate_loop, generate_chain, \
    generate_three_arms, generate_collect_stochastic, generate_bridge_stochastic, generate_river_continuous, \
    generate_race, generate_cliff, generate_trade, generate_toy, generate_gridworld, generate_mountain, \
    generate_cartpole, generate_race_full, generate_gridworld_discrete, generate_trade_discrete, generate_trade_sim


from rl.wrappers import NormalizeWrapper, ReparametrizeWrapper, PILCOWrapper, ScaleRewardWrapper, ClipRewardWrapper, \
    ScaledObservationWrapper

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)
register(
    id='FrozenLakeNotSlippery-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

game_to_env = {
    "Chain": generate_chain,
    "Taxi": generate_taxi,
    "TaxiEasy": generate_taxi_easy,
    "Loop": generate_loop,
    "Toy": generate_toy,
    "RiverSwim": generate_river,
    "SixArms": generate_arms,
    "ThreeArms": generate_three_arms,
    "Gridworld": generate_gridworld,
    "GridworldDiscrete": generate_gridworld_discrete,
    "MiniGrid-Collect-Stochastic-9x9-v0": generate_collect_stochastic,
    "Bridge-stochastic": generate_bridge_stochastic,
    "RiverSwim-continuous": generate_river_continuous,
    "RaceStrategy": generate_race,
    "Trading-v0": generate_trade,
    "Trading_discrete-v0": generate_trade_discrete,
    "Cliff": generate_cliff,
    "MountainCar": generate_mountain,
    "Cartpole": generate_cartpole,
    "RaceStrategy-v1": generate_race_full,
    "TradingSim": generate_trade_sim
}


def get_base_env(env):
    """ removes all wrappers """
    while hasattr(env, 'env'):
        env = env.env
    return env


def is_atari_game(env):
    """ Verify whether game uses the Arcade Learning Environment """
    env = get_base_env(env)
    return hasattr(env, 'ale')


def make_game(game, game_params=None):
    """ Modifications to Env """
    if game_params is None:
        game_params = {}
    if game in game_to_env.keys():
        return game_to_env[game](**game_params)

    name, version = game.rsplit('-', 1)
    if len(version) > 2:
        modify = version[2:]
        game = name + '-' + version[:2]
    else:
        modify = ''

    #print('Making game {}'.format(game))
    env = gym.make(game)
    # remove timelimit wrapper
    try:
        if type(env) == gym.wrappers.time_limit.TimeLimit:
            env = env.env
    except:
        pass

    if is_atari_game(env):
        env = prepare_atari_env(env)
    else:
        env = prepare_control_env(env, game, modify)
    return env


def prepare_control_env(env, game, modify):
    if 'n' in modify and type(env.observation_space) == gym.spaces.Box:
        print('Normalizing input space')
        env = NormalizeWrapper(env)
    if 'r' in modify:
        print('Reparametrizing the reward function')
        env = ReparametrizeWrapper(env)
    if 'p' in modify:
        env = PILCOWrapper(env)
    if 's' in modify:
        print('Rescaled the reward function')
        env = ScaleRewardWrapper(env)

    if 'CartPole' in game:
        env.observation_space = gym.spaces.Box(np.array([-4.8, -10, -4.8, -10]), np.array([4.8, 10, 4.8, 10]))
    return env


def prepare_atari_env(Env, frame_skip=3, repeat_action_prob=0.0, reward_clip=True):
    """ Initialize an Atari environment """
    env = get_base_env(Env)
    env.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_prob)
    env.frame_skip = frame_skip
    Env = ScaledObservationWrapper(Env)
    if reward_clip:
        Env = ClipRewardWrapper(Env)
    return Env

if __name__ == '__main__':
    make_game("MiniGrid-Collect-Stochastic-9x9-v0")