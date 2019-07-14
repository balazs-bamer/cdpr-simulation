import tensorflow as tf
from functools import reduce
from operator import mul


class ParameterizedFunction:
    def __init__(self, dtype, gs):
        # make one long vector of params
        all_param = tf.reshape(tf.concat([[tf.reshape(param, [-1])] for param in gs.a_param_list], name='AP', axis=1), [-1, 1])
        learnable_param = tf.reshape(tf.concat([[tf.reshape(param, [-1])] for param in gs.l_param_list], name='LP', axis=1), [-1, 1])

        # getting number of params
        numberOfAP = tf.identity(tf.constant(value=[sum([reduce(mul, param.get_shape().as_list(), 1) for param in gs.a_param_list])], dtype=tf.int32), name='numberOfAP')
        numberOfLP = tf.identity(tf.constant(value=[sum([reduce(mul, param.get_shape().as_list(), 1) for param in gs.l_param_list])], dtype=tf.int32), name='numberOfLP')

        # param assign methods
        a_param_assign_ph = tf.placeholder(dtype, name='AP_placeholder')
        l_param_assign_ph = tf.placeholder(dtype, name='LP_placeholder')
        tau = tf.placeholder(dtype, name='tau')

        a_param_assign_split = tf.split(a_param_assign_ph, [reduce(mul, param.get_shape().as_list(), 1) for param in gs.a_param_list], 1)
        l_param_assign_split = tf.split(l_param_assign_ph, [reduce(mul, param.get_shape().as_list(), 1) for param in gs.l_param_list], 1)

        ap_assign_op_list = []
        lp_assign_op_list = []
        interpolate_ap_op_list = []
        interpolate_lp_op_list = []

        for idx, param in enumerate(gs.a_param_list):
            reshaped_input_vector = tf.reshape(a_param_assign_split[idx], shape=tf.shape(param))
            ap_assign_op_list += [param.assign(reshaped_input_vector)]
            interpolate_ap_op_list += [param.assign(reshaped_input_vector * tau + param * (1 - tau))]

        for idx, param in enumerate(gs.l_param_list):
            reshaped_input_vector = tf.reshape(l_param_assign_split[idx], shape=tf.shape(param))
            lp_assign_op_list += [param.assign(reshaped_input_vector)]
            interpolate_lp_op_list += [param.assign(reshaped_input_vector * tau + param * (1 - tau))]

        all_parameters_assign_all_op = tf.group(*ap_assign_op_list, name='assignAP')
        learnable_parameters_assign_all_op = tf.group(*lp_assign_op_list, name='assignLP')

        interpolateAP_op = tf.group(*interpolate_ap_op_list, name='interpolateAP')
        interpolateLP_op = tf.group(*interpolate_lp_op_list, name='interpolateLP')

        # variables for train option
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("trainingOptions", reuse=True):
            self.max_grad_norm = tf.Variable(tf.constant(100, dtype=dtype), name='max_grad_norm', trainable=False)
            self.initial_lr = tf.Variable(tf.constant(0.001, dtype=dtype), name='InitLR', trainable=False)
            self.decayRate_lr = tf.Variable(tf.constant(1, dtype=dtype), name='DecayRateLR', trainable=False) #default = no decay
            self.decayStep_lr = tf.Variable(tf.constant(1, dtype=tf.int32), name='DecayStepLR', trainable=False)
            self.learningRate = tf.train.exponential_decay(self.initial_lr,
                                                           tf.train.get_global_step(),
                                                           self.decayStep_lr,
                                                           self.decayRate_lr,
                                                           name='LR')
            param_assign_placeholder = tf.placeholder(dtype=dtype, shape=[1, 1], name='param_assign_placeholder')
            param_assign_placeholder_int = tf.placeholder(dtype=tf.int32, shape=[1, 1], name='param_assign_placeholder_int')

        tf.assign(self.initial_lr, tf.reshape(param_assign_placeholder, []), name='InitLR_assign')
        tf.assign(self.decayRate_lr, tf.reshape(param_assign_placeholder, []), name='DecayRateLR_assign')
        tf.assign(self.decayStep_lr, tf.reshape(param_assign_placeholder_int, []), name='DecayStepLR_assign')
        tf.assign(self.max_grad_norm, tf.reshape(param_assign_placeholder, []), name='max_norm_assign')
        tf.assign(global_step, 0, name = 'reset_global_step')
        tf.assign(global_step, global_step + 1, name = 'increment_global_step')


# this class contains various methods used for learning
class SpecializedFunction(ParameterizedFunction):
    def __init__(self, dtype, gs):
        super(SpecializedFunction, self).__init__(dtype, gs)


# this class contains the actual structure of the network
class GraphStructure:
    def __init__(self, dtype):
        updateBN = tf.cast(tf.reshape(tf.placeholder(dtype=dtype, shape=[1], name='updateBNparams'), shape=[]), dtype=tf.bool)
        self.extraCost = tf.zeros([], dtype=dtype)
