import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from helpers import check_space
import os
import joblib


def OneHot(inputs, state_dim=None):
    assert state_dim is not None, "No state dimension provided for one hot encoding"
    return tf.squeeze(tf.one_hot(inputs, state_dim, axis=1), axis=2)


#### Neural Networks ##
class Model():

    def __init__(self, Env, lr, n_hidden_layers, n_hidden_units):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.state_dim, self.state_discrete = check_space(Env.observation_space)

        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        ###### Build the model ######

        self.inp = None
        self.x = None

        # Input layer
        if not self.state_discrete:
            self.inp = layers.Input(shape=np.append(None, self.state_dim), dtype="float32")
        else:
            # TODO Might be possible to use embedding here
            self.inp = layers.Input(dtype="int32", shape=np.append(None, 1))
            self.x = layers.Lambda(OneHot, arguments={"state_dim": self.state_dim})(self.inp)
            # self.x = tf.squeeze(tf.one_hot(self.inp, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc.model.py
        for i in range(n_hidden_layers):
            self.x = (layers.Dense(n_hidden_units, activation=tf.nn.elu))(self.x)

        # Output

        log_pi_hat = layers.Dense(self.action_dim, activation=None)(self.x)
        self.pi_hat = layers.Softmax()(log_pi_hat)  # policy head
        self.V_hat = layers.Dense(1, activation=None)(self.x)  # value head

        self.model = tf.keras.Model(inputs=self.inp, outputs=[self.V_hat, self.pi_hat])

        self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr),
                           loss=[keras.losses.mean_squared_error(), keras.losses.categorical_crossentropy()],
                           loss_weights=[1.0, 1.0])

        # Loss
        # self.V = tf.compat.v1.placeholder("float32", shape=[None, 1], name='V')
        # self.pi = tf.compat.v1.placeholder("float32", shape=[None, self.action_dim], name='pi')
        # self.V_loss = tf.compat.v1.losses.mean_squared_error(labels=self.V, predictions=self.V_hat)
        # self.pi_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.pi, logits=log_pi_hat)
        # self.loss = self.V_loss + tf.reduce_mean(input_tensor=self.pi_loss)
        #
        # self.lr = tf.Variable(lr, name="learning_rate", trainable=False)
        # optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr)
        # self.train_op = optimizer.minimize(self.loss)

    def train(self, sb, Vb, pib):
        targets = (Vb, pib)
        return self.model.train_on_batch(sb, y=targets)

    def predict_V(self, s):
        if len(s.shape) != len(self.x.shape):
            s = s.reshape((-1,) + s.shape)
        return self.model.predict(s)[0]

    def predict_pi(self, s):
        if len(s.shape) != len(self.x.shape):
            s = s.reshape((-1,) + s.shape)
        return self.model.predict(s)[1]

    def save(self, save_path, variables=None):
        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        pickle.dump(self.model, open(save_path, 'wb'))

    def load(self, load_path, variables=None):
        self.model = pickle.load(open(load_path, 'rb'))
