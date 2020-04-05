import numpy as np

from model_tf2 import ModelWrapper


class Wrapper(object):
    def __init__(self, root_index, mcts_maker, model_save_file, model_wrapper_params,
                 mcts_params, is_atari, n_mcts, mcts_env, c_dpw,
                 temp, game_maker=None, Env=None, mcts_only=True):

        assert game_maker is not None or Env is not None, "No environment or maker provided to the wrapper"

        self.root_index = root_index
        self.Env = Env
        self.mcts_only = mcts_only
        self.episode_probabilities = []
        self.curr_probs = None
        self.starting_states = []
        self.model_file = model_save_file
        self.model_wrapper_params = model_wrapper_params
        self.mcts_maker = mcts_maker
        self.mcts_params = mcts_params
        self.game_maker = game_maker
        self.model = None
        self.mcts = None
        #self.action_dim = Env.action_space.n
        self.is_atari = is_atari
        self.n_mcts = n_mcts
        self.mcts_env = mcts_env
        self.mcts_only = mcts_only
        self.c_dpw = c_dpw
        self.temp = temp

        if not self.is_atari:
            self.mcts_env = None

    def pi_wrapper(self, s, max_depth):
        if self.mcts_only:
            self.search(self.n_mcts, self.c_dpw, self.mcts_env, max_depth)
            state, pi, V = self.return_results(self.temp)  # TODO put 0 if the network is enabled
            self.curr_probs.append(pi)
            max_p = np.max(pi)
            a_w = np.random.choice(np.argwhere(pi == max_p)[0])
        else:
            pi_w = self.get_model().predict_pi(s).flatten()
            self.curr_probs.append(pi_w)
            max_p = np.max(pi_w)
            a_w = np.random.choice(np.argwhere(pi_w == max_p)[0])
        return a_w

    def get_env(self):
        if self.Env is None:
            self.make_env()
        return self.Env

    def get_model(self):
        if not self.model:
            pass
            #self.model = ModelWrapper(**self.model_wrapper_params)
            #self.model.load(self.model_file)
        return self.model

    def get_mcts(self):
        if not self.mcts:
            self.make_mcts()
        return self.mcts

    def make_mcts(self):
        self.mcts = self.mcts_maker(root_index=self.root_index, root=None, model=self.get_model(),
                                    na=self.Env.action_space.n, **self.mcts_params)

    def make_env(self):
        if self.game_maker is None:
            pass
        else:
            builder = self.game_maker["game_maker"]
            game = self.game_maker["game"]
            game_params = self.game_maker["game_params"]
            self.Env = builder(game, game_params)
            seed = np.random.randint(1e7)  # draw some Env seed
            self.Env.seed(seed)
            self.Env.reset()

    def reset(self):
        s = self.get_env().reset()
        self.make_mcts()
        self.starting_states.append(s)
        if self.curr_probs is not None:
            self.episode_probabilities.append(self.curr_probs)
        self.curr_probs = []
        return s

    def forward(self, a, s, r):
        if self.mcts_only:
            self.get_mcts().forward(a, s, r)

    def step(self, a):
        return self.get_env().step(a)

    def search(self, n_mcts, c_dpw, mcts_env, max_depth=200):
        self.get_mcts().search(n_mcts=n_mcts, c=c_dpw, Env=self.get_env(), mcts_env=mcts_env, max_depth=max_depth)

    def return_results(self, temp):
        return self.get_mcts().return_results(temp=temp)


class PolicyEvalWrapper(object):
    def __init__(self, env_wrapper, is_atari, n_mcts, mcts_env, c_dpw, temp, mcts_only=True):
        self.env_wrapper = env_wrapper
        self.is_atari = is_atari
        self.n_mcts = n_mcts
        self.mcts_env = mcts_env
        self.mcts_only = mcts_only
        self.c_dpw = c_dpw
        self.temp = temp

        if not self.is_atari:
            self.mcts_env = None

    def pi_wrapper(self, s):
        if self.mcts_only:
            self.env_wrapper.search(self.n_mcts, self.c_dpw, self.mcts_env)
            state, pi, V = self.env_wrapper.return_results(self.temp)  # TODO put 0 if the network is enabled
            self.env_wrapper.curr_probs.append(pi)
            max_p = np.max(pi)
            a_w = np.random.choice(np.argwhere(pi == max_p)[0])
        else:
            pi_w = self.env_wrapper.get_model().predict_pi(s).flatten()
            self.env_wrapper.curr_probs.append(pi_w)
            max_p = np.max(pi_w)
            a_w = np.random.choice(np.argwhere(pi_w == max_p))
        return a_w
