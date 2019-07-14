import functions.Policy as pc
import tensorflow as tf
import core
import Utils
from operator import mul
from functools import reduce
import numpy as np


class RecurrentStochasticPolicy(pc.Policy):
    def __init__(self, dtype, gs):
        # shortcuts
        action_dim = int(gs.output.shape[-1])
        state_dim = int(gs.input.shape[-1])

        action = tf.identity(gs.output, name=self.output_names[0])

        # standard deviation layer
        with tf.name_scope('stdevconcatOutput'):
            wo = tf.Variable(tf.zeros(shape=[1, action_dim], dtype=dtype), name='W',
                             trainable=True)  # Log standard deviation
            action_stdev = tf.identity(tf.exp(wo), name='stdev')

        gs.l_param_list.append(wo)
        gs.a_param_list.append(wo)

        super(RecurrentStochasticPolicy, self).__init__(dtype, gs)

        stdev_assign_placeholder = tf.placeholder(dtype, shape=[1, action_dim], name='Stdev_placeholder')
        Stdev_assign = tf.assign(wo, tf.log(stdev_assign_placeholder), name='assignStdev')
        Stdev_get = tf.exp(wo, name='getStdev')

        tangent_in = tf.placeholder(dtype, shape=[1, None], name='tangent')
        old_stdv = tf.placeholder(dtype, shape=[1, action_dim], name='stdv_o')
        old_action_sampled = tf.placeholder(dtype, shape=[None, None, action_dim], name='sampledAction')
        old_action_noise = tf.placeholder(dtype, shape=[None, None, action_dim], name='actionNoise')
        advantage_in = tf.placeholder(dtype, shape=[None, None], name='advantage')

        # Algorithm params
        kl_coeff = tf.Variable(tf.constant(1.0, dtype=dtype), name='kl_coeff', trainable=False)
        ent_coeff = tf.Variable(tf.constant(0.01, dtype=dtype), name='ent_coeff', trainable=False)
        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param', trainable=False)

        PPO_params_placeholder = tf.placeholder(dtype, shape=[1, 3], name='PPO_params_placeholder')

        param_assign_op_list = []
        param_assign_op_list += [
            tf.assign(kl_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 0], [1, 1]), []), name='kl_coeff_assign')]
        param_assign_op_list += [
            tf.assign(ent_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 1], [1, 1]), []), name='ent_coeff_assign')]
        param_assign_op_list += [
            tf.assign(clip_param, tf.reshape(tf.slice(PPO_params_placeholder, [0, 2], [1, 1]), []), name='clip_param_assign')]

        PPO_param_assign_ops = tf.group(*param_assign_op_list, name='PPO_param_assign_ops')

        mask = tf.sequence_mask(gs.seq_length, maxlen=tf.shape(gs.input)[1], name='mask')
        action_target = tf.placeholder(dtype, shape=[None, None, None], name='targetAction')

        action_masked = tf.boolean_mask(action, mask)
        action_target_masked = tf.boolean_mask(action_target, mask)

        with tf.name_scope('trainUsingTarget'):
            core.square_loss_opt(dtype, action_target_masked, action_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingGrad'):
            gradient_from_critic = tf.placeholder(dtype, shape=[1, None], name='Inputgradient')
            train_using_grad_optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)

            split_parameter_gradients = tf.split(gradient_from_critic,
                                                 [reduce(mul, param.get_shape().as_list(), 1) for param in
                                                  gs.a_param_list], 1)
            manipulated_parameter_gradients = []
            for grad, param in zip(split_parameter_gradients, gs.l_param_list):
                manipulated_parameter_gradients += [tf.reshape(grad, tf.shape(param))]

            manipulated_parameter_gradients_and_parameters = zip(manipulated_parameter_gradients, gs.l_param_list)
            train_using_gradients = train_using_grad_optimizer.apply_gradients(
                manipulated_parameter_gradients_and_parameters, name='applyGradients')

        util = Utils.Utils(dtype)

        with tf.name_scope('Algo'):
            logp_n = tf.boolean_mask(util.log_likelihood(action, action_stdev, old_action_sampled), mask)
            logp_old = tf.boolean_mask(util.log_likelihood(old_action_noise, old_stdv), mask)
            advantage = tf.boolean_mask(advantage_in, mask)
            ratio = tf.exp(logp_n - logp_old)
            ent = tf.reduce_sum(wo + .5 * tf.cast(tf.log(2.0 * np.pi * np.e), dtype=dtype), axis=-1)
            meanent = tf.reduce_mean(ent)

            with tf.name_scope('PPO'):
                surr1 = tf.multiply(ratio, advantage)
                surr2 = tf.multiply(tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param), advantage)
                PPO_loss = tf.reduce_mean(tf.maximum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

                kl_ = tf.boolean_mask(
                    util.kl_divergence((old_action_sampled - old_action_noise), old_stdv, action, action_stdev), mask)
                kl_mean = tf.reshape(tf.reduce_mean(kl_), shape=[1, 1, 1], name='kl_mean')

                Total_loss = PPO_loss - tf.multiply(ent_coeff, meanent)
                Total_loss2 = PPO_loss - tf.multiply(ent_coeff, meanent) + tf.multiply(kl_coeff, tf.reduce_mean(kl_))

                policy_gradient = tf.identity(tf.expand_dims(util.flatgrad(Total_loss, gs.l_param_list, self.max_grad_norm), axis=0), name='Pg')  # flatgrad
                policy_gradient2 = tf.identity(tf.expand_dims(util.flatgrad(Total_loss2, gs.l_param_list, self.max_grad_norm), axis=0), name='Pg2')  # flatgrad