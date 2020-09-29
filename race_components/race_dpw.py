import copy

from pure_mcts.keyset import KeySet
from pure_mcts.mcts_dpw import MCTSStochastic, StochasticAction, StochasticState
import numpy as np
from race_components.race_ol_uct import strategic_rollout

from helpers import (argmax, is_atari_game, copy_atari_state, restore_atari_state, stable_normalizer)


class RaceStochasticState(StochasticState):
    def __init__(self, index, r, terminal, parent_action, na, signature, budget, env=None, max_depth=200, owner=None):
        assert owner is not None, "Owner parameter must be specified for RaceStochasticState class constructor"
        self.owner = owner
        self.end_turn = env.has_transitioned()

        super(RaceStochasticState, self).__init__(index, r, terminal, parent_action, na, signature, budget,
                                                  env=env, max_depth=max_depth)
        action_list = env.get_available_actions(owner)
        self.child_actions = [RaceStochasticAction(a, parent_state=self, owner=owner) for a in action_list]

    def random_rollout(self, budget, env, max_depth=200):
        return strategic_rollout(env, budget, max_depth=200, terminal=self.terminal, root_owner=self.owner)

class RaceStochasticAction(StochasticAction):
    def __init__(self, index, parent_state, Q_init=0.0, owner=None):
        assert owner is not None, "Owner must be specified for constructor of RaceStochasticAction class"
        self.owner = owner
        super(RaceStochasticAction, self).__init__(index, parent_state, Q_init=Q_init)

    def add_child_state(self, s1, r, terminal, signature, budget, env=None, max_depth=200):
        if not env.is_terminal():
            r = r[self.owner]

        child_state = RaceStochasticState(s1, r, terminal, self, self.parent_state.na, signature, budget, env=env,
                                            max_depth=max_depth, owner=env.get_next_agent())
        self.child_states.append(child_state)

        sk = KeySet(s1)

        # s1_hash = s1.tostring()
        # self.state_indeces[sk.__hash__()] = self.n_children
        self.state_indices[sk] = self.n_children
        self.n_children += 1
        return child_state, child_state.remaining_budget

class RaceMCTSStochastic(MCTSStochastic):
    def __init__(self, root, root_index, model, na, gamma, alpha=0.6, depth_based_bias=False, owner=None):
        super(RaceMCTSStochastic, self).__init__(root, root_index, model, na, gamma,
                                                 alpha=alpha, depth_based_bias=depth_based_bias)

        assert owner is not None, "Owner must be specified for RaceMCTSStochastic constructor"
        self.owner = owner

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
            self.root = RaceStochasticState(self.root_index, r=0.0, terminal=False, parent_action=None,
                                            na=self.na, signature=Env.get_signature(),
                                            env=mcts_env, budget=budget, owner=self.owner)
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
            mcts_env.seed()
            st = 0
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias)
                k = np.ceil(self.beta * action.n ** self.alpha)
                if k >= action.n_children:
                    s1, r, t, _ = mcts_env.step(action.index)
                    # if action.index == 0 and not np.array_equal(s1.flatten(), action.parent_state.index.flatten()):
                    #     print("WTF")
                    if mcts_env.has_transitioned():
                        budget -= 1
                    if action.get_state_ind(s1) != -1:
                        state = action.child_states[action.get_state_ind(s1)]  # select
                        state.r = r
                    else:
                        state, budget = action.add_child_state(s1, r, t, mcts_env.get_signature(), budget, env=mcts_env,
                                                               max_depth=max_depth - st)  # expand
                        break
                else:
                    state = action.sample_state()
                    mcts_env.set_signature(state.signature)
                    if state.terminal:
                        budget -= 1

                if mcts_env.has_transitioned():
                    st += 1

            # Back-up
            R = np.zeros(mcts_env.agents_number)

            if not state.terminal:
                R = copy.deepcopy(state.V)
            state.update()
            agents_reward = copy.deepcopy(state.r)
            while state.parent_action is not None:  # loop back-up until root is reached
                owner = state.parent_action.owner  # rewards are stored in the state following the action, which has different owner
                if not state.terminal:
                    if state.end_turn:
                        agents_reward = copy.deepcopy(state.r)
                    R[owner] = agents_reward[owner] + self.gamma * R[owner]
                else:
                    R = copy.deepcopy(state.r)
                action = state.parent_action
                action.update(R[action.owner])
                state = action.parent_state
                state.update()