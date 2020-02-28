from six import StringIO
from gym import utils
from contextlib import closing
import sys
import numpy as np

class TaxiVisualizer(object):

    def __init__(self, grid):
        self.desc = np.array(grid)
        self.passenger_number = 0
        self.goal_number = 0
        self.passengers = np.argwhere(self.desc == 'F')
        self.passenger_number = len(self.passengers)
        self.goals = np.argwhere(self.desc == 'G')
        self.goal_number = len(self.goals)
        self.rows = self.desc.shape[0]
        self.cols = self.desc.shape[1]

        self.cells = np.argwhere(self.desc != '#')
        self.size = len(self.cells)

        self.prev_state = 0
        super().__init__()


    def reset(self):
        self.__init__(self.desc)

    def decode(self, state):
        cell_idx = int(round(state[0] * self.size))
        taxi_row, taxi_col = self.cells[cell_idx]

        if np.sum(np.abs(self.cells[cell_idx] - self.cells[self.prev_state])) > 1:
            print("Skipping cells")
            print("cell index:", state[0]*self.size)
            exit()

        self.prev_state = cell_idx

        pass_idx = np.argwhere(state[-self.passenger_number:] == 1.0).tolist()

        dest_idxs = [self.cells[int(round(state[i]*self.size))]
                     for i in range(1 + self.passenger_number, len(state) - self.passenger_number)]

        return taxi_row, taxi_col, pass_idx, dest_idxs

    def visualize_taxi(self, state, action, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        # out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(state)

        def ul(x):
            return "_" if x == " " else x

        if len(pass_idx) < self.passenger_number:
            out[taxi_row][taxi_col] = utils.colorize(
                out[taxi_row][taxi_col], 'yellow', highlight=True)
            for passenger in pass_idx:
                pi, pj = self.passengers[passenger].squeeze()
                out[pi][pj] = utils.colorize(out[pi][pj], 'blue', bold=True)
        else:  # passenger in taxi
            out[taxi_row][taxi_col] = utils.colorize(
                ul(out[taxi_row][taxi_col]), 'green', highlight=True)

        for dest in dest_idx:
            di, dj = dest
            out[di][dj] = utils.colorize(out[di][dj], 'magenta', bold="True")

        outfile.write("".join(["".join(row) + "\n" for row in out]) + "\n")

        if action is not None:
            outfile.write("({} - {})\n".format(action, ["North", "South", "West", "East", "Pickup", "Dropoff"][action]))
            outfile.write("Currently picked passengers: {}\n".format(pass_idx))

        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
