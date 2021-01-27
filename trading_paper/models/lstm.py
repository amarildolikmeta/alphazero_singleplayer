import tensorflow as tf
from trading_paper.models.models import Model
import numpy as np


class LSTM(Model):
    def __init__(
            self,
            learning_rate,
            num_layers,
            size,
            size_layer,
            output_size,
            forget_bias=0.1,
    ):
        super(LSTM, self).__init__(size, output_size)

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        self.initialize()

    def update(self, batch_x, batch_y, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")

        logits, last_state, _, loss = self.sess.run(
            [self.logits, self.last_state, self.optimizer, self.cost],
            feed_dict={
                self.X: batch_x,
                self.Y: batch_y,
                self.hidden_layer: state,
            }
        )
        return logits, last_state, loss

    def forward(self, batch_x, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")

        out_logits, last_state = self.sess.run(
            [self.logits, self.last_state],
            feed_dict={
                self.X: batch_x,
                self.hidden_layer: state,
            }
        )
        return out_logits, last_state


class BidirectionalLSTM(Model):
    def __init__(
            self,
            learning_rate,
            num_layers,
            size,
            size_layer,
            output_size,
            forget_bias=0.1,
    ):
        super(BidirectionalLSTM, self).__init__(size, output_size)

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        backward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )
        forward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple=False,
        )

        drop_backward = tf.contrib.rnn.DropoutWrapper(
            backward_rnn_cells, output_keep_prob=forget_bias
        )
        forward_backward = tf.contrib.rnn.DropoutWrapper(
            forward_rnn_cells, output_keep_prob=forget_bias
        )
        self.backward_hidden_layer = tf.placeholder(
            tf.float32, shape=(None, num_layers * 2 * size_layer)
        )
        self.forward_hidden_layer = tf.placeholder(
            tf.float32, shape=(None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(
            forward_backward,
            drop_backward,
            self.X,
            initial_state_fw=self.forward_hidden_layer,
            initial_state_bw=self.backward_hidden_layer,
            dtype=tf.float32,
        )

        self.outputs = tf.concat(self.outputs, 2)

        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        self.initialize()

    def update(self, batch_x, batch_y, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")
        assert state.shape[0] == 2, "Need both direction state"

        logits, last_state, _, loss = self.sess.run(
            [self.logits, self.last_state, self.optimizer, self.cost],
            feed_dict={
                self.X: batch_x,
                self.Y: batch_y,
                self.forward_hidden_layer: state[0],
                self.backward_hidden_layer: state[1],
            }
        )
        return logits, last_state, loss

    def forward(self, batch_x, state):
        if len(batch_x.shape) == 2:
            batch_x = np.expand_dims(
                batch_x, axis=0
            )
        elif len(batch_x.shape) != 3:
            raise Exception("Provide a window of observations")
        assert state.shape[0] == 2, "Need both direction state"

        out_logits, last_state = self.sess.run(
            [self.logits, self.last_state],
            feed_dict={
                self.X: batch_x,
                self.forward_hidden_layer: state[0],
                self.backward_hidden_layer: state[1],
            }
        )
        return out_logits, last_state


class GRU(LSTM):
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
    ):
        super(LSTM, self).__init__(size, output_size)

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.GRUCell(size_layer)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )

        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        self.initialize()


class BidirectionalGRU(BidirectionalLSTM):
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        super(BidirectionalLSTM, self).__init__(size, output_size)

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.GRUCell(size_layer)

        backward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        forward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )

        drop_backward = tf.contrib.rnn.DropoutWrapper(
            backward_rnn_cells, output_keep_prob = forget_bias
        )
        forward_backward = tf.contrib.rnn.DropoutWrapper(
            forward_rnn_cells, output_keep_prob = forget_bias
        )
        self.backward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * size_layer)
        )
        self.forward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * size_layer)
        )
        self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(
            forward_backward,
            drop_backward,
            self.X,
            initial_state_fw = self.forward_hidden_layer,
            initial_state_bw = self.backward_hidden_layer,
            dtype = tf.float32,
        )
        self.outputs = tf.concat(self.outputs, 2)
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        self.initialize()