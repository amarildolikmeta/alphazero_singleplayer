from rl.gpomdp import learn
from rl.make_game import make_game
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from rl.policies.policies_bc import build_policy
from rl.common.policies import build_policy as build_policy_trpo
import utils.tf_util as U
from rl.common.models import mlp
from gym.spaces import Discrete, Box
import argparse
import time
import numpy as np
import os
from rl.common.input import observation_placeholder
import matplotlib.pyplot as plt

def load_policy(model_path, input_dim, output_dim, num_hidden, num_layers, init_logstd=1., discrete=False,
                beta=1.0):
    observation_space = Box(low=-np.inf, high=np.inf, shape=(input_dim,))
    if discrete:
        action_space = Discrete(n=output_dim)
    else:
        action_space = Box(low=-np.inf, high=np.inf, shape=(output_dim,))
    tf.reset_default_graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
        device_count={'CPU': 8}
    )
    config.gpu_options.allow_growth = True
    sess = U.make_session(make_default=True, config=config)
    network = mlp(num_hidden=num_hidden, num_layers=num_layers)
    # policy_train = build_policy(observation_space, action_space, network, trainable_variance=True,
    #                             state_dependent_variance=True, beta=beta, init_logstd=init_logstd)()
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    policy_train = build_policy_trpo(env, network, value_network='copy')(observ_placeholder=ob)
    U.initialize()
    if model_path != '':
        policy_train.load(model_path)
    return policy_train


parser = argparse.ArgumentParser()
parser.add_argument('--game', default='RaceStrategy-v0', help='Training environment')
parser.add_argument('--grid', type=str, default="grid.txt", help='TXT file specfying the game grid')
parser.add_argument('--batch_size', type=int, default=20, help='Number of episodes in a gradient batch')
parser.add_argument('--horizon', type=int, default=100, help='steps of the horizon')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--beta', type=float, default=1., help='Temperature of the softmax policy')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount parameter')
parser.add_argument('--eval_frequency', type=int, default=20, help='Frequency of evaluations of the policy')
parser.add_argument('--eval_episodes', type=int, default=20, help='Number of episodes of evaluation')
parser.add_argument('--logdir', type=str, default='data/', help='Directory of logs')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
parser.add_argument('--n_hidden_units', type=int, default=16, help='Number of units per hidden layers in NN')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs of training')

args = parser.parse_args()
game_params = {}

# Accept custom grid if the environment requires it
if args.game == 'Taxi' or args.game == 'TaxiEasy':
    game_params['grid'] = args.grid
    game_params['box'] = True
if args.game == 'RaceStrategy-v0':
    game_params['horizon'] = args.horizon

basedir = args.logdir
ts = str(time.time())
logdir = basedir + args.game + '/' + ts + '/'
if not os.path.exists(logdir):
        os.makedirs(logdir)
env = make_game(args.game, game_params)
state_dim = env.observation_space.shape[0]

discrete = True
try:
    action_dim = env.action_space.n
except:
    action_dim = env.action_space.high.shape[0]
    discrete = False
pi = load_policy(model_path='', input_dim=state_dim, output_dim=action_dim, num_hidden=args.n_hidden_units,
                 num_layers=args.n_hidden_layers, discrete=discrete, beta=1.0)
# s = env.reset()
# start = time.time()
# for i in range(1000):
#     pi.step(s, stochastic=True)
# duration = time.time() - start
# print(duration)
optimized_policy = learn(pi=pi, env=env, max_iterations=args.n_epochs, batch_size=args.batch_size,
                         eval_frequency=args.eval_frequency, eval_episodes=args.eval_episodes, horizon=args.horizon,
                         gamma=args.gamma, logdir=logdir, lr=args.lr)

s = env.reset()
done = False

states = []
actions = []
s = 0
delta_state = 0.2
while s < env.dim[0]:
    a, _, _, _ = optimized_policy.step([s])
    states.append(s)
    actions.append(a[0])
    s += delta_state
s = env.reset()
plt.plot(states, actions)
plt.show()

while not done:
    print("State:", s)
    a, _, _, _ = optimized_policy.step(s)
    s, r, _, _ = env.step(a)
    print("Action: ", a)
    input()


print("Finished")
