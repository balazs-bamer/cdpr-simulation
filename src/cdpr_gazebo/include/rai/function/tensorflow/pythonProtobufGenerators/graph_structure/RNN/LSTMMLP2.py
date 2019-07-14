import BaseClasses as bc
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers import fully_connected


# Implementation of LSTM + MLP layers
class LSTMMLP2(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):
        super(LSTMMLP2, self).__init__(dtype)
        assert len(fn.input_names) == 2 and len(fn.output_names) == 1, "The function is not compatible with GRUMLP2"
        nonlin_str = param[0]
        nonlin = getattr(tf.nn, nonlin_str)
        weight = float(param[1])

        check=0
        for i, val in enumerate(param[2:]):
            if val == '/':
                check = i

        rnnDim = [int(i) for i in param[2:check+2]]
        mlpDim = [int(i) for i in param[check+3:]]

        self.input1 = tf.placeholder(dtype, shape=[None, None, rnnDim[0]], name=fn.input_names[0])  # [batch, time, statedim]
        self.input2 = tf.placeholder(dtype, shape=[None, None, rnnDim[1]], name=fn.input_names[1])  # [batch, time, actiondim]

        inputconcat = tf.concat([self.input1, self.input2], axis= 2, name = "inputconcat")


        length_ = tf.placeholder(dtype, name='length')  # [batch]
        length_ = tf.cast(length_, dtype=tf.int32)
        self.seq_length = tf.reshape(length_, [-1])

        # GRU
        cells = []
        state_size = []
        recurrent_state_size = 0

        for size in rnnDim[2:]:
            cell = rnn.LSTMCell(size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
            cells.append(cell)
            recurrent_state_size += cell.state_size.c + cell.state_size.h
            state_size.append(cell.state_size.c)
            state_size.append(cell.state_size.h)
        cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        hiddenStateDim = tf.identity(tf.constant(value=[recurrent_state_size], dtype=tf.int32), name='h_dim')
        h_in = tf.placeholder(dtype=dtype, shape=[None, recurrent_state_size], name='h_init')
        init_states = tf.split(h_in, num_or_size_splits=state_size, axis = 1)

        print(init_states)



        init_state_list = []
        for i in range(len(cells)):
            init_state_list.append(rnn.LSTMStateTuple(init_states[2*i], init_states[2*i+1]))
        init_state_tuple = tuple(init_state_list)

        # LSTM output
        LSTMOutput, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputconcat, sequence_length=self.seq_length, dtype=dtype, initial_state=init_state_tuple)

        # FCN
        top = tf.reshape(LSTMOutput, shape=[-1, rnnDim[-1]], name='fcIn')

        layer_n = 0
        for dim in mlpDim[:-1]:
            with tf.name_scope('hidden_layer'+repr(layer_n)):
                top = fully_connected(activation_fn=nonlin, inputs=top, num_outputs=dim, weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                layer_n += 1

        with tf.name_scope('output_layer'):
            wo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[mlpDim[-2], mlpDim[-1]], minval=-float(weight), maxval=float(weight)))
            bo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[mlpDim[-1]], minval=-float(weight), maxval=float(weight)))
            top = tf.matmul(top, wo) + bo

        self.output = tf.reshape(top, [-1, tf.shape(self.input1)[1], mlpDim[-1]])

        final_state_list = []
        for state_tuple in final_state:
            final_state_list.append(state_tuple.c)
            final_state_list.append(state_tuple.h)

        hiddenState = tf.concat([state for state in final_state_list], axis=1, name='h_state')

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
        self.net = None
