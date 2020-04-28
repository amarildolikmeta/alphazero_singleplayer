import numpy as np
from utils.adam import Adam
from gym.spaces import Box
from tqdm import trange


def eval_policy(pi, env, num_episodes=50, gamma=0.99, horizon=50, stochastic=False):
    rets = []

    num_stops = []
    avg_damages = []
    for i in range(num_episodes):
        state = env.reset()
        ret = 0
        done = False
        num_pits = 0
        avg_damage = 0
        t = 0
        while not done and t < horizon:
            a, _, _, _ = pi.step(state, stochastic=stochastic)
            state, reward, done, info = env.step(a)
            num_pits += (1 if a == 0 else 0)
            tire_damage = state[1]
            avg_damage += tire_damage
            ret += gamma ** t * reward
            t += 1
            if done:
                avg_damage /= t
                break
        rets.append(ret)
        num_stops.append(num_pits)
        avg_damages.append(avg_damage)
    return rets, num_stops, avg_damages


def create_batch_trajectories(pi,
                              env,
                              batch_size,
                              horizon,
                              render=False,
                              stochastic=True,
                              gamma=0.99):
    rets = []
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    states = np.zeros((batch_size, horizon, state_dim))
    actions = np.zeros((batch_size, horizon, action_dim))
    rewards = np.zeros((batch_size, horizon))
    for batch in range(batch_size):
        state = env.reset()
        done = False
        ret = 0
        for t in range(horizon):
            action, _, _, _ = pi.step(state, stochastic=stochastic)
            next_state, reward, done, info = env.step(action)
            states[batch, t] = state
            actions[batch, t] = action
            rewards[batch, t] = reward
            ret += gamma ** t * reward
            if done:
                # print(rewards[batch])
                # print('done')
                rewards[batch, t + 1:] = 0
                actions[batch, t + 1:] = env.action_space.sample()
                states[batch, t + 1:] = next_state

                break
            state = next_state
        rets.append(ret)
    return states, actions, rewards, rets


def learn(
          pi,
          env,
          max_iterations=1000,
          batch_size=50,
          eval_frequency=50,
          eval_episodes=50,
          horizon=50,
          gamma=0.99,
          logdir='./',
          lr=0.1):
    """

    @param pi: Policy to optimize
    @type pi: Policy
    @param env: Environment
    @type env: gym.Env
    @param max_iterations: number of gradient steps
    @type max_iterations: int
    @param batch_size: Number of trajectories in a gradient batch
    @type batch_size: int
    @param eval_frequency: frequency of evaluation of the policy
    @type eval_frequency: int
    @param eval_episodes: number of episodes of evaluation
    @type eval_episodes: int

    @param logdir: directory where to save outputs
    @type logdir: str
    @param horizon: environment horizon
    @type horizon: int
    @param gamma: discount factor
    @type gamma: float
    @param lr: learning rate
    @type lr: float
    @return: optimized policy
    @rtype: Policy
    """

    offline_scores = []
    online_scores = []
    params = pi.get_weights()
    best_eval = -np.inf
    optimizer = Adam(learning_rate=lr, ascent=True, size=params.shape[0])
    for it in trange(max_iterations):
        params = pi.get_weights()
        if it % eval_frequency == 0:
            print("Evaluating policy for %d episodes" % (eval_episodes))
            rets, stops, damages = eval_policy(pi, env, num_episodes=eval_episodes, stochastic=False, horizon=horizon)
            mean_return = np.mean(rets)
            print("Policy performance %f" % (mean_return))
            offline_scores.append([mean_return, np.mean(stops), np.mean(damages)])
            np.save(logdir + 'offline_scores.npy', offline_scores)
            pi.save(logdir + 'last')
            if mean_return > best_eval:
                pi.save(logdir + 'best')
                best_eval = mean_return

        states, actions, rewards, rets = create_batch_trajectories(pi, env, horizon=horizon, batch_size=batch_size,
                                                                   stochastic=True, gamma=gamma)
        mean_return = np.mean(rets)
        online_scores.append([mean_return, np.std(rets)])
        np.save(logdir + 'online_scores.npy', online_scores)
        grad = compute_gradient(pi, states, actions, rewards, gamma=gamma, num_params=params.shape[0])
        #grad_2 = compute_gradient_2(pi, states, actions, rewards, gamma=gamma, num_params=params.shape[0])
        #diff = grad_2 - grad
        # step = optimizer.update(grad)
        step = lr * grad
        grad_norm = np.linalg.norm(grad)
        print("Iteration %d \t Return %f \t grad_norm %f" % ((it + 1), mean_return, grad_norm))
        pi.set_weights(params + step)
    return pi


def gradient_est(pi, batch_size, len_trajectories, states, actions, num_params):
    gradients = np.zeros((batch_size, len_trajectories, num_params))
    for b in range(batch_size):
        for t in range(len_trajectories):
            action = actions[b, t]
            if np.isnan(action[0]):
                gradients[b, t, :] = np.zeros_like(gradients[b, t, :])
            else:
                state = np.array(states[b, t])
                grads = pi.compute_gradients(state, action)[0]
                gradients[b, t, :] = grads
                # gradients[b, t, :] = pi.compute_grad() (((action - np.dot(param.T, state)).reshape(-1, 1) * np.array(
                #     [state, state])).T / var_policy).reshape(-1)
    return gradients


def compute_gradient(pi, states, actions, rewards, num_params, gamma=0.99):
    batch_size, horizon, obs_dim = states.shape[:]
    discount_factor_timestep = np.power(gamma * np.ones(horizon), range(horizon))
    discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards[:, :, np.newaxis]  # (N,T,L)
    gradients = gradient_est(pi, batch_size, horizon, states, actions, num_params=num_params)  # (N,T,K, 2)
    gradient_est_timestep = np.cumsum(gradients, axis=1)  # (N,T,K, 2)
    baseline_den = np.mean(gradient_est_timestep ** 2 + 1.e-10, axis=0)  # (T,K, 2)
    baseline_num = np.mean(
        (gradient_est_timestep ** 2)[:, :, :, np.newaxis, np.newaxis] * discounted_return[:, :, np.newaxis, np.newaxis, :],
        axis=0)  # (T,K,2,L)
    baseline = baseline_num / baseline_den[:, :, np.newaxis, np.newaxis]  # (T,K,2,L)
    gradient = np.mean(np.sum(
        gradient_est_timestep[:, :, :, np.newaxis, np.newaxis] * (discounted_return[:, :, np.newaxis, np.newaxis, :] -
                                                         baseline[np.newaxis, :, :]), axis=1),
        axis=0)  # (K,2,L)
    return gradient.flatten()


def compute_gradient_2(pi, states, actions, rewards,  num_params=1, gamma=1., verbose=False, use_baseline=True):
    discount_f = gamma
    num_episodes, episode_length, state_dim = states.shape
    gamma = []
    for t in range(episode_length):
        gamma.append(discount_f ** t)
    if verbose:
        print("Episodes:", num_episodes)
    r_dim = 1
    tiled_gamma = np.tile(gamma, (r_dim, 1)).transpose()
    discounted_phi = []
    features = []
    for i in range(num_episodes):
        ft = rewards[i].reshape(episode_length, 1)
        ft = ft * tiled_gamma
        features.append(ft)
    discounted_phi = np.array(features)
    expected_discounted_phi = discounted_phi.sum(axis=1).mean(axis=0)
    print('Features Expectations:', expected_discounted_phi)
    episode_gradients = []
    probs = []

    grads = []
    for i in range(states.shape[0]):
        grads.append([])
        for j in range(states.shape[1]):
            st = states[i, j]
            action = actions[i, j]
            step_gradients, _, _, _ = pi.compute_gradients(st, action)
            if np.isnan(step_gradients).any():
                print("NAN Grad")
            grads[i].append(step_gradients.tolist())
    gradients = np.array(grads)
    # GPOMDP optimal baseline computation
    num_params = gradients.shape[2]
    cum_gradients = np.transpose(np.tile(gradients, (r_dim, 1, 1, 1)), axes=[1, 2, 3, 0]).cumsum(axis=1)

    phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1, 1)), axes=[1, 2, 0, 3])
    '''
    # Freeing memory
    del X_dataset
    del y_dataset
    del r_dataset
    del episode_gradients
    del gamma
    del discounted_phi
    '''
    # GPOMDP objective function gradient estimation
    if use_baseline:
        num = (cum_gradients ** 2 * phi).sum(axis=0)
        den = (cum_gradients ** 2).sum(axis=0) + 1e-10
        baseline = num / den
        baseline = np.tile(baseline, (num_episodes, 1, 1, 1))
        phi = phi - baseline

    estimated_gradients = (cum_gradients * (phi)).sum(axis=1)
    return np.mean(estimated_gradients, axis=0).flatten()