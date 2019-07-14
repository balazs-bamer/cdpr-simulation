import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.layers.xavier_initializer as xavier


# GRU gate with reduced state
class StateEstimatorCell(rnn.RNNCell):
    def __init__(self, dtype, output_size, state_size, hidden_layer_size, activation=tf.nn.tanh, reuse=None):
        super(StateEstimatorCell, self).__init__(_reuse=reuse)
        self._output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self._state_size = state_size
        self._activation = activation
        self._reuse = reuse
        self.dtype = dtype

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._ouput_size

    def call(self, inputs, state, scope=None):
        hidden_output = fully_connected(activation_fn=self._activation, inputs=[inputs, state], num_outputs=self.hidden_layer_size, weights_initializer=xavier, trainable=True)
        output_and_state_delta = fully_connected(activation_fn=self._activation, inputs=[hidden_output], num_outputs=self.ouput_size+self.state_size, weights_initializer=xavier, trainable=True)
        output, d = tf.split(value=output_and_state_delta, num_or_size_splits=[self.ouput_size, self.state_size], axis=1)
        new_state = state + d
        return output, new_state
