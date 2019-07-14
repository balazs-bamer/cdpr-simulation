import BaseClasses as bc
import tensorflow as tf
import core


class Vfunction(bc.SpecializedFunction):
    input_names = ['state']
    output_names = ['value']

    def __init__(self, dtype, gs):
        super(Vfunction, self).__init__(dtype, gs)

        # variables
        state_dim = gs.input.shape[1]
        state = gs.input
        value = tf.identity(gs.output, name=self.output_names[0])

        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param', trainable=False)
        clip_param_decay_rate = tf.Variable(tf.constant(1, dtype=dtype), name='clip_param_decay_rate', trainable=False)
        clip_range = tf.train.exponential_decay(clip_param,
                                                tf.train.get_global_step(),
                                                self.decayStep_lr,
                                                clip_param_decay_rate,
                                                name='clip_range')

        # new placeholders
        value_target = tf.placeholder(dtype, shape=[None, 1], name='targetValue')
        value_pred = tf.placeholder(dtype, shape=[None, 1], name='predictedValue')

        tf.identity(value_pred, name='test')

        # Assign ops.
        param_assign_placeholder = tf.placeholder(dtype=dtype, shape=[1, 1], name='param_assign_placeholder')
        tf.assign(clip_param, tf.reshape(param_assign_placeholder, []), name='clip_param_assign')
        tf.assign(clip_param_decay_rate, tf.reshape(param_assign_placeholder, []), name='clip_decayrate_assign')

        # gradients
        jac_V_wrt_State = tf.identity(tf.gradients(tf.reduce_mean(value), state)[0], name='gradient_AvgOf_V_wrt_State')

        # solvers
        with tf.name_scope('trainUsingTargetValue'):
            core.square_loss_opt(dtype, value_target, value, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingTargetValue_huber'):
            core.huber_loss_opt(dtype, value_target, value, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingTRValue'):
            core.square_loss_trust_region_opt(dtype, value_target, value, value_pred,
                                             tf.train.AdamOptimizer(learning_rate=self.learningRate),
                                             clipRange=clip_range, maxnorm=self.max_grad_norm)