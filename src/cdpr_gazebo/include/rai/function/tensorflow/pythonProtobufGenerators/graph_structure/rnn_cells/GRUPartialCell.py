import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.layers.xavier_initializer as xavier


# GRU gate with reduced state
class GRUPartialCell(rnn.RNNCell):
    def __init__(self, dtype, out_size, state_size, activation=tf.nn.tanh, reuse=None):
        self.out_size = out_size
        self.state_size = state_size
        self._activation = activation
        self._reuse = reuse
        self.dtype = dtype

    @property
    def state_size(self):
        return self.out_size

    @property
    def output_size(self):
        return self.out_size

    def __call__(self, inputs, state, scope=None):
        # We start with bias of 1.0 to not reset and not update.
        value = fully_connected(activation_fn=tf.nn.sigmoid, inputs=[inputs, state], num_outputs=2*self.out_size, weights_initializer=xavier, trainable=True)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        c = fully_connected(activation_fn=self._activation, inputs=[inputs, r*state], num_outputs=self.out_size, weights_initializer=xavier, trainable=True)
        output = u * state + (1 - u) * c
        dummy, new_state = tf.split(value=output, num_or_size_splits=[self.out_size - self.state_size, self.state_size], axis=1)
        return output, new_state
