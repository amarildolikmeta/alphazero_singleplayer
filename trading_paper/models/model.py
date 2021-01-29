import joblib
import tensorflow as tf
import numpy as np
import os


def save_variables(save_path, sess, variables=None):
    variables = variables or tf.trainable_variables()

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, sess, variables=None,  extra_vars=None):
    variables = variables or tf.trainable_variables()

    loaded_params = joblib.load(os.path.expanduser(load_path))
    if extra_vars is not None:
        for k, v in extra_vars.items():
            loaded_params[k] = v
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))
    sess.run(restores)


class Model:
    def __init__(
            self,
            size,
            output_size,
    ):
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

    def update(self, batch_x, batch_y):
        pass

    def forward(self, batch_x):
        pass

    def initialize(self):
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def update(self, batch_x, batch_y, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")

        logits, _, loss = self.sess.run(
            [self.logits, self.optimizer, self.cost],
            feed_dict={
                self.X: batch_x,
                self.Y: batch_y,
            }
        )
        return logits, state, loss

    def forward(self, batch_x, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")

        out_logits = self.sess.run(
            self.logits,
            feed_dict={
                self.X: batch_x,
            }
        )
        return out_logits, state

    def close(self):
        self.sess.close()

    def save(self, model_path):
        save_variables(model_path, sess=self.sess, variables=self.vars)

    def load(self, model_path):
        load_variables(model_path, sess=self.sess, variables=self.vars)

    def get_init_state(self):
        return None
