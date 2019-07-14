import BaseClasses as bc
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import layer_norm


# multiple gated recurrent unit layers (https://arxiv.org/pdf/1406.1078v3.pdf)
# Implementation of GRU + MLP layers
class LSTM_merged(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):
        super(LSTM_merged, self).__init__(dtype)
        nonlin_str = param[0]
        nonlin = getattr(tf.nn, nonlin_str)
        weight = float(param[1])

        check=0
        for i, val in enumerate(param[2:]):
            if val == '/':
                check = i

        rnnDim = [int(i) for i in param[2:check+2]]
        mlpDim = [int(i) for i in param[check+3:]]

        self.input = tf.placeholder(dtype, shape=[None, None, rnnDim[0]], name=fn.input_names[0])  # [batch, time, dim]

        length_ = tf.placeholder(dtype, name='length')  # [batch]
        length_ = tf.cast(length_, dtype=tf.int32)
        self.seq_length = tf.reshape(length_, [-1])

        # GRU
        cells = []
        state_size = []
        recurrent_state_size = 0
        for size in rnnDim[1:]:
            cell = rnn.LSTMCell(size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
            # cell = rnn.LayerNormBasicLSTMCell(size)
            cells.append(cell)
            recurrent_state_size += cell.state_size.c + cell.state_size.h
            state_size.append(cell.state_size.c)
            state_size.append(cell.state_size.h)

        cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        hiddenStateDim = tf.identity(tf.constant(value=[recurrent_state_size], dtype=tf.int32), name='h_dim')
        init_states = tf.split(tf.placeholder(dtype=dtype, shape=[None, recurrent_state_size], name='h_init'), num_or_size_splits=state_size, axis = 1)

        init_state_list = []
        for i in range(len(cells)):
            init_state_list.append(rnn.LSTMStateTuple(init_states[2*i], init_states[2*i+1]))
        init_state_tuple = tuple(init_state_list)

        # Full-length output for training
        gruOutput, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.input, sequence_length=self.seq_length, dtype=dtype, initial_state=init_state_tuple)
        rnn_out_flat = tf.reshape(gruOutput,shape=[-1, rnnDim[-1]], name='fcIn')

        # FCN
        # Policy
        layer_n = 0
        top = rnn_out_flat
        # top = layer_norm(rnn_out_flat)

        for dim in mlpDim[:-1]:
            with tf.name_scope('policy_hidden_layer'+repr(layer_n)):
                top = fully_connected(activation_fn=nonlin, inputs=top, num_outputs=dim, weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                layer_n += 1

        with tf.name_scope('policy_output_layer'):
            wo1 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[mlpDim[-2], mlpDim[-1]], minval=-float(weight), maxval=float(weight)))
            bo1 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[mlpDim[-1]], minval=-float(weight), maxval=float(weight)))
            top = tf.matmul(top, wo1) + bo1

        self.policyOut = tf.reshape(top, [-1, tf.shape(self.input)[1], mlpDim[-1]])

        # Value
        layer_n = 0
        top = rnn_out_flat
        # top = layer_norm(rnn_out_flat)

        for dim in mlpDim[:-1]:
            with tf.name_scope('value_hidden_layer'+repr(layer_n)):
                top = fully_connected(activation_fn=nonlin, inputs=top, num_outputs=dim, weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                layer_n += 1

        with tf.name_scope('value_output_layer'):
            wo2 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[mlpDim[-2], 1], minval=-float(weight), maxval=float(weight)))
            bo2 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[1], minval=-float(weight), maxval=float(weight)))
            top = tf.matmul(top, wo2) + bo2

        self.valueOut = tf.reshape(top, [-1, tf.shape(self.input)[1]])

        final_state_list = []
        for state_tuple in final_state:
            final_state_list.append(state_tuple.c)
            final_state_list.append(state_tuple.h)

        hiddenState = tf.concat([state for state in final_state_list], axis=1, name='h_state')

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
        print(self.a_param_list)
        self.net = None
