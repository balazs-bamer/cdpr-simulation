import BaseClasses as bc
import tensorflow as tf
import core


class Qfunction(bc.SpecializedFunction):
    input_names = ['state', 'sampledAction']
    output_names = ['QValue']

    def __init__(self, dtype, gs):
        super(Qfunction, self).__init__(dtype, gs)

        # variables
        q_value = tf.identity(gs.output, name=self.output_names[0])
        state = gs.input1
        action = gs.input2

        # new placeholders
        q_value_target = tf.placeholder(dtype, shape=[None, 1], name='targetQValue')
        tf.reduce_mean(q_value, name='average_Q_value')

        # gradients
        jac_Q_wrt_State = tf.identity(tf.gradients(tf.reduce_mean(q_value), gs.input1)[0], name='gradient_AvgOf_Q_wrt_State')
        jac_Q_wrt_Action = tf.identity(tf.gradients(tf.reduce_mean(q_value), gs.input2)[0], name='gradient_AvgOf_Q_wrt_action')

        # solvers
        with tf.name_scope('trainUsingTargetQValue'):
            core.square_loss_opt(dtype, q_value_target, q_value, tf.train.AdamOptimizer(learning_rate=self.learningRate))

        with tf.name_scope('trainUsingTargetQValue_huber'):
            core.huber_loss_opt(dtype, q_value_target, q_value, tf.train.AdamOptimizer(learning_rate=self.learningRate))
