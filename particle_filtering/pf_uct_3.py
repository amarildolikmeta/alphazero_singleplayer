import copy
from rl.make_game import is_atari_game
import numpy as np
from particle_filtering.pf_uct import PFMCTS


class PFMCTS3(PFMCTS):

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
            flag = False
            source_particle = None
            while not state.terminal:
                bias = c * self.gamma ** st / (1 - self.gamma) if self.depth_based_bias else c
                action = state.select(c=bias, variance=self.variance)
                st += 1
                k = np.ceil(self.beta * state.n ** self.alpha)
                if action.child_state is not None:
                    # select
                    could_sample = state.get_n_particles() >= k
                    state = action.child_state
                    if could_sample and source_particle is None:
                        if state.terminal:
                            source_particle, budget = action.sample_from_parent_state(mcts_env, budget)
                            state.add_particle(source_particle)
                            break
                    elif not could_sample and source_particle is None:
                        source_particle, budget = action.sample_from_parent_state(mcts_env, budget)
                        state.add_particle(source_particle)
                        if source_particle.terminal:
                            break
                    elif source_particle is not None:
                        source_particle, budget = action.sample_from_particle(source_particle, mcts_env, budget)
                        state.add_particle(source_particle)
                        if source_particle.terminal:
                            break
                    elif state.terminal:
                        source_particle = np.random.choice(state.particles)
                        budget -= 1  # sample from the terminal states particles

                else:
                    rollout_depth = max_depth if fixed_depth else max_depth - st
                    state, budget, source_particle = action.add_child_state(mcts_env, budget, max_depth=rollout_depth,
                                                                            source_particle=source_particle,
                                                                            depth=st)  # expand
                    break

            # Back-up
            R = state.V
            state.update()
            particle = source_particle
            while state.parent_action is not None:  # loop back-up until root is reached
                r = particle.reward
                if not particle.terminal:
                    R = r + self.gamma * R
                else:
                    R = r
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()
                particle = particle.parent_particle
