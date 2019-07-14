import functions.Policy as pc
import tensorflow as tf
import core
import Utils
from operator import mul
from functools import reduce
import numpy as np


class StochasticPolicy(pc.Policy):
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

        super(StochasticPolicy, self).__init__(dtype, gs)

        stdev_assign_placeholder = tf.placeholder(dtype, shape=[1, action_dim], name='Stdev_placeholder')
        Stdev_assign = tf.assign(wo, tf.log(stdev_assign_placeholder), name='assignStdev')
        Stdev_get = tf.exp(wo, name='getStdev')

        tangent_in = tf.placeholder(dtype,  name='tangent')
        old_stdv = tf.placeholder(dtype, shape=[1, action_dim], name='stdv_o')  # TODO : Change to tf.Variable
        old_action_in = tf.placeholder(dtype, name='sampledAction')
        old_action_noise_in = tf.placeholder(dtype, name='actionNoise')
        advantage_in = tf.placeholder(dtype, name='advantage')

        tangent_ = tf.reshape(tangent_in, [1, -1])
        old_action_sampled = tf.reshape(old_action_in, [-1, action_dim])
        old_action_noise = tf.reshape(old_action_noise_in, [-1, action_dim])
        advantage = tf.reshape(advantage_in, shape=[-1], name='test')

        # Algorithm params
        kl_coeff = tf.Variable(tf.constant(1.0, dtype=dtype), name='kl_coeff', trainable=False)
        ent_coeff = tf.Variable(tf.constant(0.01, dtype=dtype), name='ent_coeff', trainable=False)
        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param', trainable=False)
        clip_param_decay_rate = tf.Variable(tf.constant(1, dtype=dtype), name='clip_param_decay_rate', trainable=False)

        PPO_params_placeholder = tf.placeholder(dtype, shape=[1, 4], name='PPO_params_placeholder')

        param_assign_op_list = []
        param_assign_op_list += [
            tf.assign(kl_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 0], [1, 1]), []), name='kl_coeff_assign')]
        param_assign_op_list += [
            tf.assign(ent_coeff, tf.reshape(tf.slice(PPO_params_placeholder, [0, 1], [1, 1]), []), name='ent_coeff_assign')]
        param_assign_op_list += [
            tf.assign(clip_param, tf.reshape(tf.slice(PPO_params_placeholder, [0, 2], [1, 1]), []), name='clip_param_assign')]
        param_assign_op_list += [
            tf.assign(clip_param_decay_rate, tf.reshape(tf.slice(PPO_params_placeholder, [0, 3], [1, 1]), []), name='clip_decayrate_assign')]

        PPO_param_assign_ops = tf.group(*param_assign_op_list, name='PPO_param_assign_ops')

        action_target = tf.placeholder(dtype, shape=[None, None, action_dim], name='targetAction')
        action_target = tf.reshape(action_target, shape = [-1, action_dim])
        with tf.name_scope('trainUsingTarget'):
            core.square_loss_opt(dtype, action_target, action, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

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
                manipulated_parameter_gradients_and_parameters, name='applyGradients', global_step=tf.train.get_global_step())

        util = Utils.Utils(dtype)

        with tf.name_scope('Algo'):
            logp_n = util.log_likelihood(action, action_stdev, old_action_sampled)
            logp_old = util.log_likelihood(old_action_noise, old_stdv)
            ratio = tf.exp(logp_n - logp_old)
            ent = tf.reduce_sum(wo + .5 * tf.cast(tf.log(2.0 * np.pi * np.e), dtype=dtype), axis=-1)
            mean_ent = tf.reduce_mean(ent)

            with tf.name_scope('TRPO'):
                # Surrogate Loss
                surr = tf.reduce_mean(tf.multiply(ratio, advantage), name='loss')
                policy_gradient = tf.identity(util.flatgrad(surr, gs.l_param_list), name='Pg')  # flatgrad

                # Hessian Vector Product
                meanfixed = tf.stop_gradient(action)
                stdfixed = tf.stop_gradient(action_stdev)
                kl_ = tf.reduce_mean(util.kl_divergence(meanfixed, stdfixed, action, action_stdev))
                dkl_dth = tf.identity(util.flatgrad(kl_, gs.l_param_list))

                def getfvp(tangent):
                    temp = tf.reduce_sum(tf.multiply(dkl_dth, tangent))
                    return util.flatgrad(temp, gs.l_param_list)

                # Conjugate Gradient Descent

                out1, out2 = util.CG_tf(getfvp, tangent_, 100, 1e-15)
                Ng = tf.identity(out1, name='Cg')
                err = tf.identity(out2, name='Cgerror')

            with tf.name_scope('PPO'):
                clip_range = tf.train.exponential_decay(clip_param,
                                                        tf.train.get_global_step(),
                                                        self.decayStep_lr,
                                                        clip_param_decay_rate,
                                                        name='clip_range')
                # PPO's pessimistic surrogate (L^CLIP)
                surr1 = tf.multiply(ratio, advantage)  # negative, smaller the better
                surr2 = tf.multiply(tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range), advantage)
                PPO_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

                # KL divergence
                kl_ = util.kl_divergence((old_action_sampled - old_action_noise), old_stdv, action, action_stdev)
                kl_mean = tf.reduce_mean(kl_, name='kl_mean')

                Total_loss = PPO_loss - tf.multiply(ent_coeff, mean_ent)
                Total_loss2 = PPO_loss - tf.multiply(ent_coeff, mean_ent) + tf.multiply(kl_coeff, kl_mean)

                policy_gradient = tf.identity(util.flatgrad(Total_loss, gs.l_param_list, self.max_grad_norm), name='Pg')  # flatgrad
                policy_gradient2 = tf.identity(util.flatgrad(Total_loss2, gs.l_param_list, self.max_grad_norm), name='Pg2')  # flatgrad