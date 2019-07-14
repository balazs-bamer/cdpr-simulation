import BaseClasses as bc
import tensorflow as tf
import graph_structure.rnn_cells.StateEstimatorCell as rnn_cells


# multiple gated recurrent unit layers (https://arxiv.org/pdf/1406.1078v3.pdf)
class StateEstimatorNet(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):
        super(StateEstimatorNet, self).__init__(dtype)
        nonlin_str = param[0]
        nonlin = getattr(tf.nn, nonlin_str)
        input1_size = param[1]  # state
        input2_size = param[2]  # action
        output_size = param[1]  # state
        state_size = param[3]  # hidden state
        hidden_layer_size = param[4]

        self.input1 = tf.placeholder(dtype, shape=[None, None, input1_size], name=fn.input_names[0])  # [batch, time, dim]
        self.input2 = tf.placeholder(dtype, shape=[None, None, input2_size], name=fn.input_names[1])  # [batch, time, dim]
        self.input = tf.concat([self.input1, self.input2], axis=1)

        length_ = tf.placeholder(dtype, shape=[None, 1, 1], name='length')  # [batch, 1, 1]
        length_ = tf.cast(length_, dtype=tf.int32)
        self.seq_length = tf.reshape(length_, [-1])

        cell = rnn_cells.StateEstimatorCell(dtype, input1_size, state_size, hidden_layer_size, nonlin)
        hiddenStateDim = tf.identity(tf.reshape(tf.constant(value=[state_size], dtype=dtype), shape=[1, 1]), name='h_dim')
        unshapedOutput, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.input, sequence_length=self.seq_length, dtype=dtype)

        hiddenState = tf.concat([state for state in final_state], axis=1, name='h_state')
        self.output = tf.reshape(unshapedOutput, shape=[-1,  output_size], name=fn.output_names[0])

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
        self.net = None
