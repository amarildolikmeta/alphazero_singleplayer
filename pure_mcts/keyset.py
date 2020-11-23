import copy
import numpy as np


class KeySet(object):
    def __init__(self, state):
        self.state = copy.deepcopy(state)

        if type(state) == np.ndarray:
            self.state = {'state': state.tostring()}
        elif type(state) == list or type(state) == tuple:
            self.state = {'state': str(state)}
        elif type(state) == dict:
            for k in state.keys():
                if type(self.state[k]) == np.ndarray:
                    self.state[k] = self.state[k].tostring()
                elif type(self.state[k]) == list:
                    self.state[k] = str(self.state[k])
        else:
            print("Type {} cannot be handled by KeySet".format(type(state)))
            raise TypeError

    def __hash__(self):
        return hash(tuple(sorted(self.state.items())))

    def __eq__(self, other):
        print(self.state)
        if not hasattr(other, 'state'):
            return False

        if not type(other.state) == type(self.state):
            return False

        for k in self.state:
            if not (k in other.state and other.state[k] == self.state[k]):
                return False

        return True


if __name__ == '__main__':
    h = KeySet([1, 0.5, 2])
    h.__hash__()