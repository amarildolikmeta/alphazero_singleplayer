import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helpers import check_space
import os
import joblib

#### Neural Networks ##
class Model():

    def __init__(self, Env, lr, n_hidden_layers, n_hidden_units):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.state_dim, self.state_discrete = check_space(Env.observation_space)
        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        # Placeholders
        if not self.state_discrete:
            self.x = x = tf.placeholder("float32", shape=np.append(None, self.state_dim), name='x')  # state
        else:
            self.x = x = tf.placeholder("int32", shape=np.append(None, 1))  # state
            x = tf.squeeze(tf.one_hot(x, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc.
        for i in range(n_hidden_layers):
            x = slim.fully_connected(x, n_hidden_units, activation_fn=tf.nn.elu)

        # Output
        log_pi_hat = slim.fully_connected(x, self.action_dim, activation_fn=None)
        self.pi_hat = tf.nn.softmax(log_pi_hat)  # policy head
        self.V_hat = slim.fully_connected(x, 1, activation_fn=None)  # value head

        # Loss
        self.V = tf.placeholder("float32", shape=[None, 1], name='V')
        self.pi = tf.placeholder("float32", shape=[None, self.action_dim], name='pi')
        self.V_loss = tf.losses.mean_squared_error(labels=self.V, predictions=self.V_hat)
        self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pi, logits=log_pi_hat)
        self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)

        self.lr = tf.Variable(lr, name="learning_rate", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sb, Vb, pib):
        self.sess.run(self.train_op, feed_dict={self.x: sb,
                                                self.V: Vb,
                                                self.pi: pib})

    def predict_V(self, s):
        if len(s.shape) != len(self.x.shape):
            s = s.reshape((-1,) + s.shape)
        return self.sess.run(self.V_hat, feed_dict={self.x: s})

    def predict_pi(self, s):
        if len(s.shape) != len(self.x.shape):
            s = s.reshape((-1,) + s.shape)
        return self.sess.run(self.pi_hat, feed_dict={self.x: s})

    def save(self, save_path, variables=None):
        sess = self.sess
        variables = variables or tf.trainable_variables()

        ps = sess.run(variables)
        save_dict = {v.name: value for v, value in zip(variables, ps)}
        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(save_dict, save_path)

    def load(self, load_path, variables=None):
        sess = self.sess
        variables = variables or tf.trainable_variables()

        loaded_params = joblib.load(os.path.expanduser(load_path))
        restores = []
        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                restores.append(v.assign(loaded_params[v.name]))
        sess.run(restores)

    def evaluate_loss(self, episode, sb, Vb, pib):
        V_loss, pi_loss = self.sess.run([self.V_loss, self.pi_loss],
                                              feed_dict={self.x: sb,
                                                         self.V: Vb,
                                                         self.pi: pib})

        pi_loss = np.mean(pi_loss)

        # print("Episode {0:3d}:\t V Loss={1:.5f},\tpi_loss={1:.5f}".
        #      format(episode, V_loss, pi_loss))

        return V_loss, pi_loss

