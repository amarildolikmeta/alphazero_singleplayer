import numpy as np


class Adam(object):
    def __init__(self, size, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, ascent=True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.learning_rate = learning_rate
        self.dir = (1 if ascent else -1 )

    def update(self, globalg):
        self.t += 1
        stepsize = self.learning_rate
        a = stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = self.dir * a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
