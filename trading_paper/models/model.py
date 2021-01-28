import tensorflow as tf
import numpy as np


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
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def update(self, batch_x, batch_y):
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
        return logits, loss

    def forward(self, batch_x):
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
        return out_logits

    def close(self):
        self.sess.close()