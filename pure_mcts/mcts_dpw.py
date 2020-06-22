import numpy as np
import copy
from helpers import (is_atari_game, copy_atari_state, restore_atari_state, stable_normalizer, max_Q)
from pure_mcts.keyset import KeySet
from pure_mcts.mcts import MCTS, Action, State

DESTROY = True


class StochasticAction(Action):
    ''' StochasticAction object '''

    def __init__(self, index, parent_state, Q_init=0.0):
        super(StochasticAction, self).__init__(index, parent_state, Q_init)
        self.child_states = []
        self.n_children = 0
        self.state_indeces = {}

    def add_child_state(self, s1, r, terminal, signature, budget, env=None, max_depth=200):
        child_state = StochasticState(s1, r, terminal, self, self.parent_state.na, signature, budget, env=env,
                                      max_depth=max_depth)
        self.child_states.append(child_state)

        sk = KeySet(s1)

        # s1_hash = s1.tostring()
        self.state_indeces[sk] = self.n_children
        self.n_children += 1
        return child_state, child_state.remaining_budget

    def get_state_ind(self, s1):
        # s1_hash = s1.tostring()
        sk = KeySet(s1)
        try:
            index = self.state_indeces[sk]
            return index
        except KeyError:
            return -1

    def sample_state(self):
        p = []
        for i, s in enumerate(self.child_states):
            s = self.child_states[i]
            p.append(s.n / self.n)
        return self.child_states[np.random.choice(a=self.n_children, p=p)]


class StochasticState(State):
    ''' StochasticState object '''

    def __init__(self, index, r, terminal, parent_action, na, signature, budget, env=None, max_depth=200):
        super().__init__(index, r, terminal, parent_action, na, budget, env=env, max_depth=max_depth)
        self.signature = signature
        # self.priors = model.predict_pi(index).flatten()
        self.child_actions = [StochasticAction(a, parent_state=self, Q_init=self.V) for a in range(na)]


class MCTSStochastic(MCTS):
    ''' MCTS object '''

    def __init__(self, root, root_index, na, gamma, alpha=0.6, model=None):
        super(MCTSStochastic, self).__init__(root, root_index, na, gamma)
        self.alpha = alpha

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200):
        ''' Perform the MCTS search from the root '''
        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env)  # for Atari: snapshot the root at the beginning
        else:
            mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
        # else:
        #     restore_atari_state(mcts_env, snapshot)

        # Check that the environment has been copied correctly
        try:
            sig1 = mcts_env.get_signature()
            sig2 = Env.get_signature()
            if sig1.keys() != sig2.keys():
                raise AssertionError
            if not all(np.array_equal(sig1[key], sig2[key]) for key in sig1):
                raise AssertionError
        except AssertionError:
            print("Something wrong while copying the environment")
            sig1 = mcts_env.get_signature()
            sig2 = Env.get_signature()
            print(sig1.keys(), sig2.keys())
            exit()

        if self.root is None:
            # initialize new root
            self.root = StochasticState(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na,
                                        signature=Env.get_signature(), env=mcts_env, budget=budget)
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)
            # obs1 = mcts_env._get_obs().flatten()
            # obs2 = Env._get_obs().flatten()
            # if not np.array_equal(obs1, obs2):
            #     print("HOLDUP")
            mcts_env.seed()
            st = 0
            while not state.terminal:
                # obs = mcts_env._get_obs().flatten()
                # flattened_State = state.index.flatten()
                # if not np.array_equal(flattened_State, obs):
                #     print("WHATTTTTT")
                action = state.select(c=c)
                st += 1
                k = np.ceil(c * action.n ** self.alpha)
                if k >= action.n_children:
                    s1, r, t, _ = mcts_env.step(action.index)
                    # if action.index == 0 and not np.array_equal(s1.flatten(), action.parent_state.index.flatten()):
                    #     print("WTF")
                    budget -= 1
                    if action.get_state_ind(s1) != -1:
                        state = action.child_states[action.get_state_ind(s1)]  # select
                        state.r = r
                        # if state.terminal:
                        #     budget -= 1
                    else:
                        # if action.index == 0 and len(action.child_states) > 0:
                        #     print("Error")
                        state, budget = action.add_child_state(s1, r, t, mcts_env.get_signature(), budget, env=mcts_env,
                                                               max_depth=max_depth - st)  # expand
                        break
                else:
                    state = action.sample_state()
                    mcts_env.set_signature(state.signature)
                    if state.terminal:
                        budget -= 1
                    # obs = mcts_env._get_obs().flatten()
                    # flattened_State = state.index.flatten()
                    # if not np.array_equal(flattened_State, obs):
                    #     print("WHATTTTTT")

            # Back-up
            R = state.V
            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                if not state.terminal:
                    R = state.r + self.gamma * R
                else:
                    R = state.r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def return_results(self, temp, on_visits=False):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if on_visits:
            pi_target = stable_normalizer(counts, temp)
        else:
            pi_target = max_Q(Q)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        # V_target = np.max((counts / np.sum(counts)) * Q)[None]
        # V_target = np.max(Q)[None]
        return self.root.index, pi_target, V_target

    def forward(self, a, s1, r):
        ''' Move the root forward '''
        action = self.root.child_actions[a]
        if not DESTROY and action.n_children > 0:
            if action.get_state_ind(s1) == -1:
                self.root = None
            else:
                self.root = action.child_states[action.get_state_ind(s1)]
                self.root.parent_action = None
                self.root.r = r
        else:
            self.root = None
        self.root_index = s1

    def inorderTraversal(self, root, g, vertex_index, parent_index, v_label, a_label):
        if root:
            g.add_vertex(vertex_index)
            # v_label.append(str(root.index))
            v_label.append(root.to_json())
            if root.parent_action:
                g.add_edge(parent_index, vertex_index)
                a_label.append(str(root.parent_action.index) + "(" + str(root.parent_action.n) + ")")
            par_index = vertex_index
            vertex_index += 1
            for i, a in enumerate(root.child_actions):
                for s in a.child_states:
                    vertex_index = self.inorderTraversal(s, g, vertex_index, par_index, v_label, a_label)
        return vertex_index

    def print_index(self):
        print(self.count)
        self.count += 1

    def print_tree(self, root):
        self.print_index()
        for i, a in enumerate(root.child_actions):
            if hasattr(a, 'child_state'):
                self.print_tree(a.child_state)
