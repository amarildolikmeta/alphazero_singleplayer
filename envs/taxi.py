import numpy as np
from sklearn.utils.extmath import cartesian
import gym.spaces as spaces
from envs.FiniteMDP import FiniteMDP


class Taxi(FiniteMDP):
    def __init__(self, p, rew, grid_map, cell_list, passenger_list, mu=None, gamma=.9, horizon=np.inf, box=False):
        super(Taxi, self).__init__(p, rew, mu=mu, gamma=gamma, horizon=horizon)
        self.box = box
        self.cell_list = cell_list
        self.passenger_list = passenger_list
        self.passenger_indexes = [np.where((np.array(cell_list) == passenger_list[i]).all(axis=1))[0] for i in
                                  range(len(passenger_list))]
        self.passenger_states = cartesian([[0, 1]] * len(passenger_list))
        goals = np.argwhere(np.array(grid_map) == 'G')
        self.goal_indexes = [np.where((np.array(cell_list) == goals[i]).all(axis=1))[0] for i in range(len(goals))]
        if box:
            # state =[my_position, [passenger_positions], goal_position, [passenger_states]]
            self.observation_space = spaces.Box(low=np.zeros(1 + len(goals) + 2 * len(passenger_list)),
                                                high=np.ones(1 + len(goals) + 2 * len(passenger_list)))
            # env._step = env.step
            # env._reset = env.reset
            #
            #
            #
            # def step(a):
            #     s, reward, absorbing, info = env._step(a)
            #     state = index_to_box(s)
            #     return np.array(state), reward, absorbing, info
            #
            # def reset(s=None):
            #     s = env._reset(s)
            #     state = index_to_box(s)
            #     return np.array(state)
            #
            # env.step = step
            # env.reset = reset

    def step(self, action):
        s, reward, absorbing, info = super(Taxi, self).step(action)
        state = self.index_to_box(s)

        return np.array(state), reward, absorbing, info

    def reset(self, s=None):
        s = super(Taxi, self).reset(s)
        state = self.index_to_box(s)
        return np.array(state)

    def index_to_box(self, s):
        pos = s % len(self.cell_list)
        st = self.cell_list[pos[0]]
        current_passenger_state = np.zeros(len(self.passenger_list))

        idx = s // len(self.cell_list)
        current_passenger_state[:] = self.passenger_states[idx]
        state = np.array(pos.tolist() + np.array(self.passenger_indexes).flatten().tolist() + \
                         np.array(self.goal_indexes).flatten().tolist() + current_passenger_state.flatten().tolist())
        if (current_passenger_state != 0).any():
            pass
        # normalize indexes
        state[:-len(self.passenger_list)] /= len(self.cell_list)
        return state


def generate_taxi(grid, prob=.9, rew=(0, 1, 3, 15), gamma=.99, horizon=np.inf, box=False, easy_mode=False):
    """
    This Taxi generator requires a .txt file to specify the shape of the grid
    world and the cells. There are five types of cells: 'S' is the starting
    where the agent is; 'G' is the goal state; '.' is a normal cell; 'F' is a
    passenger, when the agent steps on a hole, it picks up it.
    '#' is a wall, when the agent is supposed to step on a wall, it actually
    remains in its current state. The initial states distribution is uniform
    among all the initial states provided. The episode terminates when the agent
    reaches the goal state. The reward is always 0, except for the goal state
    where it depends on the number of collected passengers. Each action has
    a certain probability of success and, if it fails, the agent goes in a
    perpendicular direction from the supposed one.

    The grid is expected to be rectangular.

    This problem is inspired from:
    "Bayesian Q-Learning". Dearden R. et al.. 1998.

    Args:
        grid (str): the path of the file containing the grid structure;
        prob (float, .9): probability of success of an action;
        rew (tuple, (0, 1, 3, 15)): rewards obtained in goal states;
        gamma (float, .99): discount factor;
        horizon (int, np.inf): the horizon.

    Returns:
        A FiniteMDP object built with the provided parameters.

    """

    if easy_mode:
        rew = (1, 3, 15, 0)

    grid_map, cell_list, passenger_list = parse_grid(grid)

    assert len(rew) == len(np.argwhere(np.array(grid_map) == 'F')) + 1

    p = compute_probabilities(grid_map, cell_list, passenger_list, prob)

    if easy_mode:
        r = compute_easy_reward(grid_map, cell_list, passenger_list, rew)
    else:
        r = compute_reward(grid_map, cell_list, passenger_list, rew)
    mu = compute_mu(grid_map, cell_list, passenger_list)

    return Taxi(p=p, mu=mu, rew=r, horizon=horizon, gamma=gamma, box=box, grid_map=grid_map,
                passenger_list=passenger_list, cell_list=cell_list)
    # env = FiniteMDP(p, r, mu, gamma, horizon)
    #
    # if box:
    #     passenger_indexes = [np.where((np.array(cell_list) == passenger_list[i]).all(axis=1))[0] for i in
    #                          range(len(passenger_list))]
    #     passenger_states = cartesian([[0, 1]] * len(passenger_list))
    #     goals = np.argwhere(np.array(grid_map) == 'G')
    #     goal_indexes = [np.where((np.array(cell_list) == goals[i]).all(axis=1))[0] for i in range(len(goals))]
    #     passenger_states = cartesian([[0, 1]] * len(passenger_list))
    #     # state =[my_position, [passenger_positions], goal_position, [passenger_states]]
    #     env.observation_space = spaces.Box(low=np.zeros(1 + len(goals) + 2 * len(passenger_list)),
    #                                        high=np.ones(1 + len(goals) + 2 * len(passenger_list)))
    #     env._step = env.step
    #     env._reset = env.reset
    #
    #     def index_to_box(s):
    #         pos = s % len(cell_list)
    #         st = cell_list[pos[0]]
    #         current_passenger_state = np.zeros(len(passenger_list))
    #
    #         idx = s // len(cell_list)
    #         current_passenger_state[:] = passenger_states[idx]
    #         state = np.array(pos.tolist() + np.array(passenger_indexes).flatten().tolist() +\
    #                 np.array(goal_indexes).flatten().tolist() + current_passenger_state.flatten().tolist())
    #         if (current_passenger_state != 0).any():
    #             pass
    #         #normalize indexes
    #         state[:-len(passenger_list)] /= len(cell_list)
    #         return state
    #
    #     def step(a):
    #         s, reward, absorbing, info = env._step(a)
    #         state = index_to_box(s)
    #         return np.array(state), reward, absorbing, info
    #
    #     def reset(s=None):
    #         s = env._reset(s)
    #         state = index_to_box(s)
    #         return np.array(state)
    #
    #     env.step = step
    #     env.reset = reset
    # return env


def parse_grid(grid):
    """
    Parse the grid file:

    Args:
        grid (str): the path of the file containing the grid structure.

    Returns:
        A list containing the grid structure.

    """
    grid_map = list()
    cell_list = list()
    passenger_list = list()
    with open(grid, 'r') as f:
        m = f.read()

        assert 'S' in m and 'G' in m

        row = list()
        row_idx = 0
        col_idx = 0
        for c in m:
            if c in ['#', '.', 'S', 'G', 'F']:
                row.append(c)
                if c in ['.', 'S', 'G', 'F']:
                    cell_list.append([row_idx, col_idx])
                    if c == 'F':
                        passenger_list.append([row_idx, col_idx])
                col_idx += 1
            elif c == '\n':
                grid_map.append(row)
                row = list()
                row_idx += 1
                col_idx = 0
            else:
                raise ValueError('Unknown marker.')

    return grid_map, cell_list, passenger_list


def compute_probabilities(grid_map, cell_list, passenger_list, prob):
    """
    Compute the transition probability matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells;
        prob (float): probability of success of an action.

    Returns:
        The transition probability matrix;

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2 ** len(passenger_list)
    p = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    passenger_states = cartesian([[0, 1]] * len(passenger_list))

    for i in range(n_states):
        idx = i // len(cell_list)
        collected_passengers = np.array(
            passenger_list)[np.argwhere(passenger_states[idx] == 1).ravel()]
        state = c[i % len(cell_list)]

        if g[tuple(state)] in ['.', 'S', 'F']:
            if g[tuple(state)] in ['F'] \
                    and state.tolist() not in collected_passengers.tolist():
                continue
            for a in range(len(directions)):
                new_state = state + directions[a]

                j = np.where((c == new_state).all(axis=1))[0]
                if j.size > 0:
                    assert j.size == 1

                    if g[tuple(new_state)] == 'F' and new_state.tolist() \
                            not in collected_passengers.tolist():
                        current_passenger_state = np.zeros(len(passenger_list))
                        current_passenger_idx = np.where(
                            (new_state == passenger_list).all(axis=1))[0]
                        current_passenger_state[current_passenger_idx] = 1
                        new_passenger_state = passenger_states[
                                                  idx] + current_passenger_state
                        new_idx = np.where((
                                passenger_states == new_passenger_state).all(
                            axis=1))[0]

                        j += len(cell_list) * new_idx
                    else:
                        j += len(cell_list) * idx
                else:
                    j = i

                p[i, a, j] = prob

                for d in [1 - np.abs(directions[a]),
                          np.abs(directions[a]) - 1]:
                    slip_state = state + d
                    k = np.where((c == slip_state).all(axis=1))[0]
                    if k.size > 0:
                        assert k.size == 1

                        if g[tuple(slip_state)] == 'F' and slip_state.tolist() \
                                not in collected_passengers.tolist():
                            current_passenger_state = np.zeros(
                                len(passenger_list))
                            current_passenger_idx = np.where(
                                (slip_state == passenger_list).all(axis=1))[0]
                            current_passenger_state[current_passenger_idx] = 1
                            new_passenger_state = passenger_states[
                                                      idx] + current_passenger_state
                            new_idx = np.where((
                                    passenger_states == new_passenger_state).all(
                                axis=1))[0]

                            k += len(cell_list) * new_idx
                        else:
                            k += len(cell_list) * idx
                    else:
                        k = i

                    p[i, a, k] += (1. - prob) * .5

    return p


def compute_reward(grid_map, cell_list, passenger_list, rew):
    """
    Compute the reward matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells;
        rew (tuple): rewards obtained in goal states.

    Returns:
        The reward matrix.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2 ** len(passenger_list)
    r = np.zeros((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    passenger_states = cartesian([[0, 1]] * len(passenger_list))

    for goal in np.argwhere(g == 'G'):
        for a in range(len(directions)):
            prev_state = goal - directions[a]
            if prev_state in c:
                for i in range(len(passenger_states)):
                    i_idx = np.where((c == prev_state).all(axis=1))[0] + len(
                        cell_list) * i
                    j_idx = j = np.where((c == goal).all(axis=1))[0] + len(
                        cell_list) * i

                    r[i_idx, a, j_idx] = rew[np.sum(passenger_states[i])]

    return r


def compute_easy_reward(grid_map, cell_list, passenger_list, rew):
    """
    Compute the reward matrix.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells;
        rew (tuple): rewards obtained in goal states.

    Returns:
        The reward matrix.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2 ** len(passenger_list)
    r = - np.ones((n_states, 4, n_states))
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    passenger_states = cartesian([[0, 1]] * len(passenger_list))

    for goal in np.argwhere(g == 'G'):
        for a in range(len(directions)):
            prev_state = goal - directions[a]
            if prev_state in c:
                for i in range(len(passenger_states)):
                    i_idx = np.where((c == prev_state).all(axis=1))[0] + len(
                        cell_list) * i
                    j_idx = j = np.where((c == goal).all(axis=1))[0] + len(
                        cell_list) * i

                    r[i_idx, a, j_idx] = 0

    for k in range(len(passenger_list)):
        passenger = np.array(passenger_list[k])

        for a in range(len(directions)):
            prev_state = passenger - directions[a]

            if prev_state in c:

                for i in range(len(passenger_states)):
                    i_idx = np.where((c == prev_state).all(axis=1))[0] + len(
                        cell_list) * i
                    j_idx = j = np.where((c == passenger).all(axis=1))[0] + len(
                        cell_list) * i

                    r[i_idx, a, j_idx] = 10

                    # If we already picked the passenger, we won't give any reward

                    # if passenger_states[i][k] == 1:
                    #     r[i_idx, a, j_idx] = 0
                    # else:
                    #     r[i_idx, a, j_idx] = rew[np.sum(passenger_states[i])]


    return r


def compute_mu(grid_map, cell_list, passenger_list):
    """
    Compute the initial states distribution.

    Args:
        grid_map (list): list containing the grid structure;
        cell_list (list): list of non-wall cells;
        passenger_list (list): list of passenger cells.

    Returns:
        The initial states distribution.

    """
    g = np.array(grid_map)
    c = np.array(cell_list)
    n_states = len(cell_list) * 2 ** len(passenger_list)
    mu = np.zeros(n_states)
    starts = np.argwhere(g == 'S')

    for s in starts:
        i = np.where((c == s).all(axis=1))[0]
        mu[i] = 1. / len(starts)

    return mu

if __name__ == '__main__':

    mdp = generate_taxi('../grid.txt', box=True, easy_mode=True)
    from utils.visualization.taxi import TaxiVisualizer

    with open("../grid.txt", 'r') as f:
        m = f.readlines()
        matrix = []
        for r in m:
            row = []
            for ch in r.strip('\n'):
                row.append(ch)
            matrix.append(row)
        visualizer = TaxiVisualizer(matrix)
        f.close()

    n_episodes = 10
    for ep in range(n_episodes):
        done = False
        s = mdp.reset()
        visualizer.reset()
        t = 0
        while not done:
            a = int(input())
            s, r, done, _ = mdp.step(a)
            visualizer.visualize_taxi(s,a)
            t += 1
