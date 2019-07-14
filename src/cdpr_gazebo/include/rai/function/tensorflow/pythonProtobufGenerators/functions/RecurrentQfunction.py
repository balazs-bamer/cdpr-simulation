import BaseClasses as bc
import tensorflow as tf
import core


class RecurrentQfunction(bc.SpecializedFunction):
    input_names = ['state', 'sampledAction']
    output_names = ['QValue']

    def __init__(self, dtype, gs):
        super(RecurrentQfunction, self).__init__(dtype, gs)
        # variables
        q_value = tf.squeeze(gs.output, axis=2, name=self.output_names[0])
        state = gs.input1
        action = gs.input2

        # new placeholders
        q_value_target = tf.placeholder(dtype, shape=[None, None], name='targetQValue') #[batch, time, 1]
        mask = tf.sequence_mask(gs.seq_length, maxlen=tf.shape(gs.input1)[1], name='mask')

        q_value_target_masked = tf.reshape(tf.boolean_mask(q_value_target, mask), [-1, 1], name = 'test')
        q_value_masked = tf.reshape(tf.boolean_mask(q_value, mask), [-1, 1])
        avg = tf.reduce_mean(q_value_masked, name='average_Q_value')

        # gradients
        jac_Q_wrt_State = tf.identity(tf.gradients(avg, gs.input1)[0], name='gradient_AvgOf_Q_wrt_State')
        jac_Q_wrt_Action = tf.identity(tf.gradients(avg, gs.input2)[0], name='gradient_AvgOf_Q_wrt_action')

        # solvers
        with tf.name_scope('trainUsingTargetQValue'):
            core.square_loss_opt(dtype, q_value_target_masked, q_value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)

        with tf.name_scope('trainUsingTargetQValue_huber'):
            core.huber_loss_opt(dtype, q_value_target_masked, q_value_masked, tf.train.AdamOptimizer(learning_rate=self.learningRate), maxnorm=self.max_grad_norm)
