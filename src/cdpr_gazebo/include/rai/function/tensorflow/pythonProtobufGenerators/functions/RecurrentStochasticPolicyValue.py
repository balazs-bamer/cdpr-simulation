import functions.Policy as pc
import tensorflow as tf
import core
import Utils
from operator import mul
from functools import reduce
import numpy as np


class RecurrentStochasticPolicyValue(pc.Policy):
    def __init__(self, dtype, gs):
        # shortcuts
        action_dim = int(gs.policyOut.shape[-1])
        state_dim = int(gs.input.shape[-1])

        action = tf.identity(gs.policyOut, name="action") ## 3D
        value = tf.identity(gs.valueOut, name="value") ## 2D

        # standard deviation layer
        with tf.name_scope('stdevconcatOutput'):
            wo = tf.Variable(tf.zeros(shape=[1, action_dim], dtype=dtype), name='W',
                             trainable=True)  # Log standard deviation
            action_stdev = tf.identity(tf.exp(wo), name='stdev')

        gs.l_param_list.append(wo)
        gs.a_param_list.append(wo)

        super(RecurrentStochasticPolicyValue, self).__init__(dtype, gs)

        stdev_assign_placeholder = tf.placeholder(dtype, shape=[1, action_dim], name='Stdev_placeholder')
        Stdev_assign = tf.assign(wo, tf.log(stdev_assign_placeholder), name='assignStdev')
        Stdev_get = tf.exp(wo, name='getStdev')

        # Algorithm placeholders
        old_stdv = tf.placeholder(dtype, shape=[1, action_dim], name='stdv_o')
        old_action_sampled = tf.placeholder(dtype, shape=[None, None, action_dim], name='sampledAction')
        old_action_noise = tf.placeholder(dtype, shape=[None, None, action_dim], name='actionNoise')
        advantage_in = tf.placeholder(dtype, shape=[None, None], name='advantage')
        value_target = tf.placeholder(dtype, shape=[None, None], name='targetValue')
        value_pred = tf.placeholder(dtype, shape=[None, None], name='predictedValue')

        # Algorithm params
        v_coeff = tf.Variable(tf.constant(0.5, dtype=dtype), name='v_coeff')
        ent_coeff = tf.Variable(tf.constant(0.01, dtype=dtype), name='ent_coeff')
        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param')
        clip_param_decay_rate = tf.Variable(tf.constant(1, dtype=dtype), name='clip_param_decay_rate')

        # Assign ops.
        PPO_params_placeholder = tf.placeholder(dtype=dtype, shape=[1, 4], name='PPO_params_placeholder')

        param_assign_op_list = []
        param_assign_op_list += [
            tf.assign(v_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 0], [1, 1]), []), name='kl_coeff_assign')]
        param_assign_op_list += [
            tf.assign(ent_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 1], [1, 1]), []), name='ent_coeff_assign')]
        param_assign_op_list += [
            tf.assign(clip_param, tf.reshape(tf.slice(PPO_params_placeholder, [0, 2], [1, 1]), []), name='clip_param_assign')]
        param_assign_op_list += [
            tf.assign(clip_param_decay_rate, tf.reshape(tf.slice(PPO_params_placeholder, [0, 3], [1, 1]), []), name='clip_decayrate_assign')]

        PPO_param_assign_ops = tf.group(*param_assign_op_list, name='PPO_param_assign_ops')

        with tf.name_scope('trainUsingGrad'):
            gradient_from_critic = tf.placeholder(dtype, shape=[1, None], name='Inputgradient')
            train_using_grad_optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate, epsilon=1e-5)
            # train_using_grad_optimizer = tf.train.RMSPropOptimizer(learning_rate=train_using_critic_learning_rate)

            split_parameter_gradients = tf.split(gradient_from_critic,
                                                 [reduce(mul, param.get_shape().as_list(), 1) for param in
                                                  gs.a_param_list], 1)
            manipulated_parameter_gradients = []
            for grad, param in zip(split_parameter_gradients, gs.l_param_list):
                manipulated_parameter_gradients += [tf.reshape(grad, tf.shape(param))]

            manipulated_parameter_gradients_and_parameters = zip(manipulated_parameter_gradients, gs.l_param_list)
            train_using_gradients = train_using_grad_optimizer.apply_gradients(
                manipulated_parameter_gradients_and_parameters, name='applyGradients', global_step=tf.train.get_global_step())
        util = Utils.Utils(dtype)

        with tf.name_scope('Algo'):
            mask = tf.sequence_mask(gs.seq_length, maxlen=tf.shape(gs.input)[1], name='mask')
            logp_n = tf.boolean_mask(util.log_likelihood(action, action_stdev, old_action_sampled), mask)
            logp_old = tf.boolean_mask(util.log_likelihood(old_action_noise, old_stdv), mask)
            advantage = tf.boolean_mask(advantage_in, mask)
            ratio = tf.exp(logp_n - logp_old)
            ent = tf.reduce_sum(wo + .5 * tf.cast(tf.log(2.0 * np.pi * np.e), dtype=dtype), axis=-1)
            meanent = tf.reduce_mean(ent)


            with tf.name_scope('RPPO'):
                clip_range = tf.train.exponential_decay(clip_param,
                                                        tf.train.get_global_step(),
                                                        self.decayStep_lr,
                                                        clip_param_decay_rate,
                                                        name='clip_range')
                #POLICY LOSS
                surr1 = tf.multiply(ratio, advantage)
                surr2 = tf.multiply(tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range), advantage)
                PPO_loss = tf.reduce_mean(tf.maximum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

                kl_ = tf.boolean_mask(
                    util.kl_divergence((old_action_sampled - old_action_noise), old_stdv, action, action_stdev), mask)
                kl_mean = tf.reshape(tf.reduce_mean(kl_), shape=[], name='kl_mean')

                #VALUE LOSS
                vpredclipped = value_pred + tf.clip_by_value(value - value_pred, tf.negative(clip_param), clip_param)
                vf_err1 = tf.square(value - value_target)
                vf_err2 = tf.square(vpredclipped - value_target)
                vf_err1_masked = tf.boolean_mask(vf_err1, mask)
                vf_err2_masked = tf.boolean_mask(vf_err2, mask)

                vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_err1_masked, vf_err2_masked))

                Total_loss = tf.identity(PPO_loss - tf.multiply(ent_coeff, meanent) + v_coeff * vf_loss, name='loss')
                policy_gradient = tf.identity(tf.expand_dims(util.flatgrad(Total_loss, gs.l_param_list, self.max_grad_norm), axis=0), name='Pg')  # flatgrad

                testTensor = tf.identity(surr1, name='test')
