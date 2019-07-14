import BaseClasses as bc
import tensorflow as tf
import core


class RecurrentVfunction(bc.SpecializedFunction):
    input_names = ['state']
    output_names = ['value']

    def __init__(self, dtype, gs):
        super(RecurrentVfunction, self).__init__(dtype, gs)
        # variables
        value = tf.squeeze(gs.output, axis=2, name=self.output_names[0])

        state = gs.input

        clip_param = tf.Variable(tf.constant(0.2, dtype=dtype), name='clip_param', trainable=False)
        clip_param_decay_rate = tf.Variable(tf.constant(1, dtype=dtype), name='clip_param_decay_rate', trainable=False)
        clip_range = tf.train.exponential_decay(clip_param,
                                                tf.train.get_global_step(),
                                                self.decayStep_lr,
                                                clip_param_decay_rate,
                                                name='clip_range')
        # new placeholders
        value_target = tf.placeholder(dtype, shape=[None, None], name='targetValue')
        value_pred = tf.placeholder(dtype, shape=[None, None], name='predictedValue')
        mask = tf.sequence_mask(gs.seq_length, maxlen=tf.shape(gs.input)[1], name='mask')
        value_target_masked = tf.boolean_mask(value_target, mask)
        value_pred_masked = tf.boolean_mask(value_pred, mask)
        value_masked = tf.boolean_mask(value, mask)

        # Assign ops.
        param_assign_placeholder = tf.placeholder(dtype, shape=[1, 1], name='param_assign_placeholder')
        tf.assign(clip_param, tf.reshape(param_assign_placeholder, []), name='clip_param_assign')
        tf.assign(clip_param_decay_rate, tf.reshape(param_assign_placeholder, []), name='clip_decayrate_assign')

        # solvers
        with tf.name_scope('trainUsingTargetValue'):
            core.square_loss_opt(dtype, value_target_masked, value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingTargetValue_huber'):
            core.huber_loss_opt(dtype, value_target_masked, value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingTRValue'):
            core.square_loss_trust_region_opt(dtype, value_target_masked, value_masked, value_pred_masked,
                                             tf.train.AdamOptimizer(learning_rate=self.learningRate),
                                             clipRange=clip_range, maxnorm=self.max_grad_norm)