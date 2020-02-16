import copy
import numpy as np
from mushroom.utils.table import EnsembleTable
import json
import os
from mcts_dpw import MCTSStochastic, StochasticState
from rl.make_game import make_game
from helpers import is_atari_game, store_safely, Database, copy_atari_state, restore_atari_state, argmax
from policies.eval_policy import eval_policy
import time

class EnvEvalWrapper(object):
    pass


class MCTSAgent(MCTSStochastic):
    def __init__(self, na, ns, alpha=1.1, beta=5., lamb=1, mu=0., gamma=0.9999, model=None):
        #self.mdp_info = mdp_info
        self.na = na
        self.ns = ns
        self.root = None
        self.gamma = gamma
        if model is None:
            self.model = self
            self.dist = np.zeros(shape=(ns, na, 4))
            for state in range(self.ns):
                for action in range(self.na):
                    self.dist[state][action][0] = alpha  # 3.
                    self.dist[state][action][1] = beta  # 3.
                    self.dist[state][action][2] = mu  # 1.5  #alpha>1 ensures the normal-gamma dist is well defined
                    self.dist[state][action][3] = lamb  # 0.75 #high beta to increase the variance of the prior distribution to explore more
            # alpha = np.ones(mdp_info.n_states) * alpha
            # beta = np.ones(mdp_info.n_states) * beta
            # mu = np.ones(mdp_info.n_states) * mu
            # lamb = np.ones(mdp_info.n_states) * lamb
            # self.dist = np.stack([alpha, beta, mu, lamb], axis=0)
            # self.rho = np.ones((mdp_info.n_states, mdp_info.n_actions, mdp_info.n_states))

            def predict(s, a):
                return self.dist[s, a]  #, self.rho[s]
            self.predict = predict

    def update(self, state, action, r, terminal=False):
        s = state.index
        if terminal:
            for action in state.child_actions:
                a = action.index
                dist_params = self.dist[s, a].flatten()
                alpha, beta, mu, lamb = dist_params[:]
                dist_params[0] = alpha + 0.5
                dist_params[1] = beta + (lamb * (0 - mu) ** 2 / (lamb + 1)) / 2
                dist_params[2] = (lamb * mu + 0) / (lamb + 1)
                dist_params[3] = lamb + 1
                self.dist[s, a] = dist_params[:]
                action.Q = dist_params[:]
                action.n += 1
            state.n += 1
            return
        a = action.index
        dist_params = self.dist[s, a].flatten()
        alpha, beta, mu, lamb = dist_params[:]
        dist_params[0] = alpha + 0.5
        dist_params[1] = beta + (lamb * (r - mu) ** 2 / (lamb + 1)) / 2
        dist_params[2] = (lamb * mu + r) / (lamb + 1)
        dist_params[3] = lamb + 1
        self.dist[s, a] = dist_params[:]
        action.Q = dist_params[:]
        action.n += 1
        state.n += 1
        #self.rho[s, a, s1] = self.rho[s, a, s1] + 1

    def rollout(self,  env, depth, H=10):
        R = 0
        while depth < H:
            s, r, t, _ = env.step(np.random.choice(self.na))
            R += r
            if t:
                break
            depth += 1
        return R

    def reset(self, root_index):
        self.root = None
        self.root_index = root_index

    def search_iteration(self, mcts_env, state, depth, H=10):
        if state.terminal:
            self.update(state, None, 0,  True)
            return 0
        elif depth >= H:
            return state.child_actions[np.random.choice(self.na)].q(stochastic=True)
        else:
            action = state.select(True)
            s1, r, t, _ = mcts_env.step(action.index)
            if action.get_state_ind(s1) == -1:
                if len(action.child_states) > 0:
                    print("New state")
                next_state = action.add_child_state(s1, r, t, self)#, mcts_env.get_signature()
                if t:
                    R = 0
                else:
                    R = next_state.child_actions[np.random.choice(self.na)].q(stochastic=True)
                next_state.n +=1
                r = r + self.gamma * R
                self.update(state, action, r)
                return r
            else:
                next_state = action.child_states[action.get_state_ind(s1)]
                next_state.r = r
                R = r + self.gamma * self.search_iteration(mcts_env, next_state, depth+1, H)
                self.update(state, action, R)
                return R

    def search(self, n_mcts, Env, mcts_env, H=30):

        ''' Perform the MCTS search from the root '''
        if self.root is None:
            # initialize new root
            self.root = ThompsonSamplingState(self.root_index, r=0.0, terminal=False, parent_action=None,
                                              na=self.na, model=self)#, signature=mcts_env.get_signature()
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning

        for i in range(n_mcts):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)

            depth = 0
            mcts_env.seed()
            self.search_iteration(mcts_env, state, depth, H)


    def return_results(self):
        ''' Process the output at the root node '''
        # counts = np.array([child_action.n for child_action in self.root.child_actions])
        # Q = np.array([child_action.Q for child_action in self.root.child_actions])
        # pi_target = stable_normalizer(counts, temp)
        # V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index.flatten(), self.root.select(False).index

    def forward(self, a, s1, r):
        ''' Move the root forward '''
        action = self.root.child_actions[a]
        if action.n_children > 0:
            if action.get_state_ind(s1) == -1:
                self.root = None
                self.root_index = s1
            else:
                self.root = action.child_states[action.get_state_ind(s1)]
                self.root.r = r
        else:
            self.root = None
            self.root_index = s1


    def learn(self,
              env,
              n_ep=1000,
              n_mcts=10,
              max_ep_len=100,
              lr=0.01,
              c=1.5,
              gamma=0.9999,
              data_size=1000,
              batch_size=32,
              temp=1.,
              n_hidden_layers=2,
              n_hidden_unit=3,
              eval_freq=-1,
              eval_episodes=100,
              alpha=0.6,
              out_dir='../',
              pre_process=None,
              visualize=False):
        ''' Outer training loop '''
        if pre_process is not None:
            pre_process()

        # tf.reset_default_graph()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        episode_returns = []  # storage
        timepoints = []
        # Environments
        #Env = make_game(game)
        is_atari = is_atari_game(env)
        #mcts_env = make_game(game) if is_atari else None
        mcts_emv = None
        online_scores = []
        offline_scores = []
        mcts_params = dict(
            gamma=gamma
        )


        # D = Database(max_size=data_size, batch_size=batch_size)
        # model = Model(Env=Env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
        t_total = 0  # total steps
        R_best = -np.Inf
        for ep in range(n_ep):
            if eval_freq > 0 and ep % eval_freq == 0 and ep > 0:  # and ep > 0
                print('Evaluating policy for {} episodes!'.format(eval_episodes))
                seed = np.random.randint(1e7)  # draw some Env seed
                env.seed(seed)
                s = env.reset()
                self.reset(s)
                #mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim, **mcts_params)
                env_wrapper = EnvEvalWrapper()
                env_wrapper.mcts = self
                starting_states = []

                def reset_env():
                    s = env.reset()
                    env_wrapper.mcts = self
                    self.reset(s)
                    #mcts_maker(root_index=s, root=None, model=model,na=model.action_dim, **mcts_params)
                    starting_states.append(s)
                    if env_wrapper.curr_probs is not None:
                        env_wrapper.episode_probabilities.append(env_wrapper.curr_probs)
                    env_wrapper.curr_probs = []
                    return s

                def forward(a, s, r):
                    env_wrapper.mcts.forward(a, s, r)
                    # pass

                env_wrapper.reset = reset_env
                env_wrapper.step = lambda x: env.step(x)
                env_wrapper.forward = forward
                env_wrapper.episode_probabilities = []
                env_wrapper.curr_probs = None

                def pi_wrapper(ob):
                    if not is_atari:
                        mcts_env = None
                    env_wrapper.mcts.search(n_mcts=n_mcts, Env=env, mcts_env=mcts_env)
                    state, a = env_wrapper.mcts.return_results()
                    # pi = model.predict_pi(s).flatten()
                    #env_wrapper.curr_probs.append(pi)
                    #a = np.argmax(pi)
                    return a

                rews, lens = eval_policy(pi_wrapper, env_wrapper, n_episodes=eval_episodes, verbose=True,
                                         max_len=max_ep_len)
                offline_scores.append([np.min(rews), np.max(rews), np.mean(rews), np.std(rews),
                                       len(rews), np.mean(lens)])
                # if len(rews) < eval_episodes or len(rews) == 0:
                #     print("WTF")
                # if np.std(rews) == 0.:
                #     print("WTF 2")
                np.save(out_dir + '/offline_scores.npy', offline_scores)
            start = time.time()
            s = env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            env.seed(seed)
            self.reset(s)
            if is_atari:
                mcts_env.reset()
                mcts_env.seed(seed)
            if ep % eval_freq == 0:
                print("Collecting %d episodes" % eval_freq)
            #mcts = mcts_maker(root_index=s, root=None, model=model, na=model.action_dim,
             #                 **mcts_params)  # the object responsible for MCTS searches
            for t in range(max_ep_len):
                # MCTS step
                if not is_atari:
                    mcts_env = None
                self.search(n_mcts=n_mcts, Env=env, mcts_env=mcts_env)  # perform a forward search
                if visualize:
                    self.visualize()
                state, a = self.return_results()  # extract the root output

                # Make the true step
                #a = np.random.choice(len(pi), p=pi)
                #a_store.append(a)
                s1, r, terminal, _ = env.step(a)
                R += r
                t_total += n_mcts  # total number of environment steps (counts the mcts steps)

                if terminal:
                    break
                else:
                    self.forward(a, s1, r)

            # Finished episode
            episode_returns.append(R)  # store the total episode return
            online_scores.append(R)
            timepoints.append(t_total)  # store the timestep count of the episode return
            store_safely(out_dir, 'result', {'R': episode_returns, 't': timepoints})
            np.save(out_dir + '/online_scores.npy', online_scores)
            # print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),
            #                                                                          np.round((time.time() - start),
            #                                                                                   1)))

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R

            # Train
            # D.reshuffle()
            # try:
            #     for epoch in range(1):
            #         for sb, Vb, pib in D:
            #             model.train(sb, Vb, pib)
            # except Exception as e:
            #     print("ASD")
            # model.save(out_dir + 'model')
        # Return results
        return episode_returns, timepoints, R_best, offline_scores


class ThompsonSamplingState(StochasticState):

    def __init__(self, index, r, terminal, parent_action, na, model):#, signature
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        #self.signature = signature
        # Child actions
        self.na = na
        #self.priors = model.predict_pi(index).flatten()
        self.child_actions = [ThompsonSamplingAction(a, parent_state=self, Q_init=self.model.predict(self.index, a))
                              for a in range(na)]

    def to_json(self):
        inf = {}
        inf["state"] = str(self.index)
        #inf["V"] = str(self.V)
        inf["n"] = self.n
        inf["terminal"] = self.terminal
        #inf["priors"] = str(self.priors)
        inf["r"] = self.r
        return json.dumps(inf)

    def select(self, stochastic=True):
        qs = np.zeros(self.na)
        for a in self.child_actions:
            qs[a.index] = a.q(stochastic)
        return self.child_actions[argmax(qs)]

    # def q(self, a, depth, stochastic, dist_params, rho):
    #     # r = 0
    #     # if stochastic:
    #     #     weights = np.random.dirichlet(rho[a])
    #     # else:
    #     #     weights = rho[a] / np.sum(rho[a], axis=-1)
    #     #
    #     # for next_state in range(self.n_states):
    #     #     r += weights[next_state] * self.v(next_state, depth, stochastic)
    #     # r = self.r * self.gamma * r
    #     # return r
    #     if stochastic:
    #         mu, tau = self.sampleNG(a.Q)
    #         return mu
    #     else:
    #         return a.Q[2]
    #
    # def v(self, s, depth, stochastic):
    #     dist_params, _ = self.model.predict(s)
    #     if depth >= self.H or self.terminal:
    #         return 0
    #     if stochastic:
    #         mu, tau = self.sampleNG(dist_params)
    #         return mu
    #     else:
    #         return dist_params[2]

    def sampleNG(self, alpha, beta, mu, lamb):
        tau = np.random.gamma(alpha, beta)
        R = np.random.normal(mu, 1.0 / (lamb * tau))
        return R, tau

class ThompsonSamplingAction:
    ''' ThompsonSamplingAction object '''

    def __init__(self, index, parent_state, Q_init):
        self.index = index
        self.parent_state = parent_state
        self.child_states = []
        self.n_children = 0
        self.state_indeces = {}
        self.W = 0.0
        self.n = 0
        self.Q = Q_init.flatten()

    def add_child_state(self, s1, r, terminal, model):#, signature
        child_state = ThompsonSamplingState(s1, r, terminal, self, self.parent_state.na, model)#, signature
        self.child_states.append(child_state)
        s1_hash = s1.tostring()
        self.state_indeces[s1_hash] = self.n_children
        self.n_children += 1
        return child_state

    def get_state_ind(self, s1):
        s1_hash = s1.tostring()
        try:
            index = self.state_indeces[s1_hash]
            return index
        except KeyError:
            return -1

    def q(self, stochastic):
        if stochastic:
            mu, tau = self.sampleNG(self.Q.flatten())
            return mu
        else:
            return self.Q[2]

    def sampleNG(self, dist_params):
        alpha, beta, mu, lamb = dist_params[:]
        tau = np.random.gamma(alpha, beta)
        R = np.random.normal(mu, 1.0 / (lamb * tau))
        return R, tau

    def sample_state(self):
        p = []
        for i, s in enumerate(self.child_states):
            s = self.child_states[i]
            p.append(s.n / self.n)
        return self.child_states[np.random.choice(a=self.n_children, p=p)]
