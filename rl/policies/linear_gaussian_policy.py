import numpy as np


class LinearGaussianPolicy:

    def __init__(self, weights=None, bias=None, noise=None):
        if weights is not None:
            self.bias = bias
            self.weights = weights
            self.output, self.input = self.weights.shape
        if noise is not None and isinstance(noise, (int,  float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        if noise is None:
            noise = np.diag(np.ones(self.output)*0)
        self.noise = noise

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, noise=None):
        self.weights = weights
        self.output, self.input = self.weights.shape
        if noise is not None and isinstance(noise, (int, float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        self.noise = noise

    def _add_noise(self):
        noise = np.random.multivariate_normal(np.zeros(self.output), self.noise, 1).T
        return noise

    def act(self, X, stochastic=True):
        X = X.reshape(self.input, 1)
        y = np.dot(self.weights, X)
        if self.bias:
            y += self.bias
        if self.noise is not None and stochastic:
            y += self._add_noise()
        return y

    def step(self, X, stochastic=False):
        return None, self.act(X, stochastic), None, None

    def compute_gradients(self, X, y, diag=False):
        X = np.array(X).reshape(self.input, 1)
        y = np.array(y).reshape(self.output, 1)
        mu = np.dot(self.weights, X)
        if self.bias:
            mu += self.bias
        if diag:
            grad = np.diag((np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))))
        else:
            grad = (np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))).flatten()
        if self.bias:
            grad = np.concatenate([grad, [1]])
        return [grad], None, None, None, None


    def compute_gradients_imp(self, X, y, pi2, diag=False):
        X = np.array(X).reshape(self.input, 1)
        y = np.array(y).reshape(self.output, 1)
        mu = np.dot(self.weights, X)
        if self.bias:
            mu += self.bias
        if diag:
            grads = np.diag((np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))))
            grads = np.dot(np.exp(-(X-pi2.weights)**2/2)/np.sqrt(2*np.pi)/pi2.noise, grads)
        else:
            grads = (np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))).flatten()
            grads = np.dot(np.exp(-(X-pi2.weights)**2/2)/np.sqrt(2*np.pi)/pi2.noise, grads)
        return grads


class LinearBoltzmanPolicy:

    def __init__(self, weights=None, bias=None, noise=None):
        if weights is not None:
            self.bias = bias
            self.weights = weights
            self.input, self.output = self.weights.shape
        if noise is not None and isinstance(noise, (int,  float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        if noise is None:
            noise = np.diag(np.ones(self.output)*0)
        self.noise = noise

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, noise=None):
        self.weights = weights
        self.input, self.output = self.weights.shape
        if noise is not None and isinstance(noise, (int, float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        self.noise = noise



    def act(self, X, stochastic=True):
        X = X.reshape(self.input, 1)
        softmax = self.policy(X, self.weights)
        if stochastic:
            return np.argmax(softmax)
        else:
            return  np.random.choice(len(softmax),p=softmax)

    def step(self, X, stochastic=False):
        return None, self.act(X, stochastic), None, None

    def policy(self, state, w):
        z = np.dot(state, w)
        shiftx = z - np.max(z)
        exp = np.exp(shiftx)
        return exp / np.sum(exp)

    def compute_gradients(self, X, y, diag=False):
        X = np.array(X).reshape(1, self.input)
        action = int(np.asscalar(np.array(y)))
        #y = np.squeeze(np.eye(self.weights.shape[0])[y])

        #mu = np.dot(self.weights, X)
        softmax = self.policy(X, self.weights)


        grad = self.softmax_grad(softmax)[action, :]
        grad = grad / softmax[0, action]
        # if diag:
        #     grad = np.diag((np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))))
        # else:
        #     grad = (np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))).flatten()
        grad = X.T.dot(grad[None, :])
        if self.bias:
            grad = np.concatenate([grad, [1]])

        epsilon = 1e-4
        w1 = np.copy(self.weights)
        w2 = np.copy(self.weights)
        w1[0, 0] += epsilon
        w2[0, 0] -= epsilon
        grad_check = (np.log(self.policy(X, w1))[0, action] -
                             np.log(self.policy(X, w2)[0, action])) / (2. * epsilon)
        assert np.isclose(grad_check, grad[0, 0])
        return [grad], softmax[:,action], None, softmax

    def softmax_grad(self, softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
