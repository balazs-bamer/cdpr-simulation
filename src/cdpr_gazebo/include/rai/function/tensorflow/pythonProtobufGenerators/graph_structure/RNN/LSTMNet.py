import BaseClasses as bc
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import fully_connected


# Implementation of LSTM + MLP layers
class LSTMNet(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):
        super(LSTMNet, self).__init__(dtype)
        nonlin_str = param[0]
        nonlin = getattr(tf.nn, nonlin_str)
        weight = float(param[1])

        dimension = [int(i) for i in param[2:]]

        self.input = tf.placeholder(dtype, shape=[None, None, dimension[0]], name=fn.input_names[0])  # [batch, time, dim]

        length_ = tf.placeholder(dtype, name='length')  # [batch]
        length_ = tf.cast(length_, dtype=tf.int32)
        self.seq_length = tf.reshape(length_, [-1])

        # GRU
        cells = []
        state_size = [] # size of c = h in LSTM
        recurrent_state_size = 0

        for size in dimension[1:-1]:
            cell = rnn.LSTMCell(size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
            # cell = rnn.LayerNormBasicLSTMCell(size, layer_norm=True)
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

        # LSTM output
        LSTMOutput, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.input, sequence_length=self.seq_length, dtype=dtype, initial_state=init_state_tuple)

        top = tf.reshape(LSTMOutput,shape=[-1, dimension[-2]], name='fcIn')

        with tf.name_scope('output_layer'):
            wo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[dimension[-2], dimension[-1]], minval=-float(weight), maxval=float(weight)))
            bo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[dimension[-1]], minval=-float(weight), maxval=float(weight)))
            top = tf.matmul(top, wo) + bo

        self.output = tf.reshape(top, [-1, tf.shape(self.input)[1], dimension[-1]])

        final_state_list = []
        for state_tuple in final_state:
            final_state_list.append(state_tuple.c)
            final_state_list.append(state_tuple.h)

        hiddenState = tf.concat([state for state in final_state_list], axis=1, name='h_state')

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
