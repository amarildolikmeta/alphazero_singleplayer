import copy
from rl.make_game import is_atari_game
import numpy as np
from particle_filtering.pf_uct import PFMCTS


class PFMCTS2(PFMCTS):

    def search(self, n_mcts, c, Env, mcts_env, budget, max_depth=200, fixed_depth=True):
        """ Perform the MCTS search from the root """
        env = copy.deepcopy(Env)
        self.create_root(env, budget)
        if self.root.terminal:
            raise (ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(env)
        if is_atari:
            raise NotImplementedError
        while budget > 0:
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                raise NotImplementedError
            mcts_env.seed(np.random.randint(1e7))
            st = 0
            terminal = False
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                st += 1
                k = np.ceil(self.beta * action.n ** self.alpha)
                if action.child_state is not None:
                    state = action.child_state  # select
                    add_particle = k >= state.get_n_particles()
                    if add_particle:
                        source_particle, budget = action.sample_from_parent_state(mcts_env, budget)
                        state.add_particle(source_particle)
                        if source_particle.terminal:
                            terminal = True
                            break
                    else:
                        particle = state.sample_reward()
                        if state.terminal or particle.terminal:
                            terminal = True
                            budget -= 1  # sample from the terminal states particles

                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, source_particle = action.add_child_state(mcts_env, budget, max_depth=rollout_depth,
                                                                            depth=st)  # expand
                    terminal = source_particle.terminal
                    break


            # Back-up
            R = state.V
            state.update()
            while state.parent_action is not None:  # loop back-up until root is reached
                r = state.reward
                if not terminal:
                    R = r + self.gamma * R
                else:
                    R = r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()
