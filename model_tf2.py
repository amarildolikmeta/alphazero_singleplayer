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
class Model(object):

    def __init__(self, Env, lr, n_hidden_layers, n_hidden_units, joint_networks=False):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.state_dim, self.state_discrete = check_space(Env.observation_space)

        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        # Build the model

        self.joint_model = joint_networks

        self.model = None
        self.value_model = None
        self.policy_model = None

        if joint_networks:
            self.model = self.build_joint_model(lr, n_hidden_layers, n_hidden_units, self.state_dim, self.action_dim, self.state_discrete)

        else:
            self.policy_model = self.build_policy_network(lr, n_hidden_layers, n_hidden_units, self.state_dim, self.action_dim, self.state_discrete)
            self.value_model = self.build_value_network(lr, n_hidden_layers, n_hidden_units, self.state_dim, self.state_discrete)

    def build_joint_model(self, lr, n_hidden_layers, n_hidden_units, state_dim, action_dim, state_discrete):
        if not state_discrete:
            inp = x = layers.Input(shape=state_dim, dtype="float32")
        else:
            # TODO Might be possible to use embedding here
            inp = layers.Input(dtype="int32", shape=(1,))
            x = layers.Lambda(OneHot, arguments={"state_dim": state_dim})(inp)
            # self.x = tf.squeeze(tf.one_hot(self.inp, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc
        for i in range(n_hidden_layers):
            x = (layers.Dense(n_hidden_units, activation=tf.nn.elu))(x)

        # Output

        log_pi_hat = layers.Dense(action_dim, activation=None)(x)
        pi_hat = layers.Softmax()(log_pi_hat)  # policy head
        V_hat = layers.Dense(1, activation=None)(x)  # value head

        model = tf.keras.Model(inputs=inp, outputs=[V_hat, pi_hat])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=[keras.losses.mean_squared_error, keras.losses.categorical_crossentropy],
                      loss_weights=[1.0, 1.0])

        return model

    def build_policy_network(self, lr, n_hidden_layers, n_hidden_units, state_dim, action_dim, state_discrete):
        lr = 0.1

        if not state_discrete:
            inp = x = layers.Input(shape=state_dim, dtype="float32")
        else:
            # TODO Might be possible to use embedding here
            inp = layers.Input(dtype="int32", shape=(1,))
            x = layers.Lambda(OneHot, arguments={"state_dim": state_dim})(inp)
            # self.x = tf.squeeze(tf.one_hot(self.inp, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc
        for i in range(n_hidden_layers):
            x = (layers.Dense(n_hidden_units, activation=tf.nn.elu))(x)

        # Output

        log_pi_hat = layers.Dense(action_dim, activation=None)(x)
        pi_hat = layers.Softmax()(log_pi_hat)  # policy head

        model = tf.keras.Model(inputs=inp, outputs=pi_hat)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=keras.losses.categorical_crossentropy)

        return model

    def build_value_network(self, lr, n_hidden_layers, n_hidden_units, state_dim, state_discrete):
        if not state_discrete:
            inp = x = layers.Input(shape=state_dim, dtype="float32")
        else:
            # TODO Might be possible to use embedding here
            inp = layers.Input(dtype="int32", shape=(1,))
            x = layers.Lambda(OneHot, arguments={"state_dim": state_dim})(inp)
            # self.x = tf.squeeze(tf.one_hot(self.inp, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc
        for i in range(n_hidden_layers):
            x = (layers.Dense(n_hidden_units, activation=tf.nn.elu))(x)

        # Output

        V_hat = layers.Dense(1, activation=None)(x)  # value head

        model = tf.keras.Model(inputs=inp, outputs=V_hat)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=keras.losses.mean_squared_error)

        return model

    def train(self, sb, Vb, pib):

        if self.joint_model:
            targets = (Vb, pib)
            return self.model.train_on_batch(sb, y=targets)

        else:
            pi_loss = self.policy_model.train_on_batch(sb, y=pib)
            v_loss = self.value_model.train_on_batch(sb, y=Vb)
            return [pi_loss + v_loss, v_loss, pi_loss]

    def predict_V(self, s):
        if len(s.shape) != self.state_dim:
            s = s.reshape((-1,) + s.shape)

        if self.joint_model:
            return self.model.predict(s)[0]
        else:
            return self.value_model.predict(s)

    def predict_pi(self, s):
        if len(s.shape) != self.state_dim:
            s = s.reshape((-1,) + s.shape)
        if self.joint_model:
            return self.model.predict(s)[1]
        else:
            return self.policy_model.predict(s)

    def save(self, save_path, variables=None):
        if not self.joint_model:
            raise NotImplementedError
        dirname = os.path.dirname(save_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        pickle.dump(self.model, open(save_path, 'wb'))

    def load(self, load_path, variables=None):
        if not self.joint_model:
            raise NotImplementedError
        self.model = pickle.load(open(load_path, 'rb'))
