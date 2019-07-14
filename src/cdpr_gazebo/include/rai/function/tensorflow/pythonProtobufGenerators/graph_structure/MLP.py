import BaseClasses as bc
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class MLP(bc.GraphStructure):
    def __init__(self, dtype, *param, fn):

        super(MLP, self).__init__(dtype)
        assert len(fn.input_names) == 1 and len(fn.output_names) == 1, "The function is not compatible with MLP"

        # params
        nonlin_str = param[0]
        weight = float(param[1])
        dimension = [int(i) for i in param[2:]]
        nonlin = getattr(tf.nn, nonlin_str)

        # input
        self.input = tf.placeholder(dtype, name=fn.input_names[0])
        self.input = tf.reshape(self.input, [-1, dimension[0]]) # reshape must be done

        # network
        top = self.input
        layer_n = 0

        for dim in dimension[1:-1]:
            with tf.name_scope('hidden_layer'+repr(layer_n)):
                top = fully_connected(activation_fn=nonlin, inputs=top, num_outputs=dim, weights_initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                layer_n += 1

        with tf.name_scope('output_layer'):
            wo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[dimension[-2], dimension[-1]], minval=-float(weight), maxval=float(weight)))
            bo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[dimension[-1]], minval=-float(weight), maxval=float(weight)))
            top = tf.matmul(top, wo) + bo

        self.output = tf.identity(top)

        self.l_param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.a_param_list = self.l_param_list
