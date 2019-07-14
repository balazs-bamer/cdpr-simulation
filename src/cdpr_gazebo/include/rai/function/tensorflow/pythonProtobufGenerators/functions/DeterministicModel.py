import BaseClasses as bc
import tensorflow as tf
import core


class DeterministicModel(bc.SpecializedFunction):

    input_names = ['input']
    output_names = ['output']

    def __init__(self, dtype, gs):
        super(DeterministicModel, self).__init__(dtype, gs)

        # shortcuts
        output_dim = int(gs.output.shape[1])
        output = tf.identity(gs.output, name=self.output_names[0])
        input = gs.input

        # jacobian
        jac_Action_wrt_State = tf.identity(tf.stack([tf.gradients(output[:, idx], input) for idx in range(output_dim)], axis=3)[0, :, :, :], name='jac_output_wrt_input')

        # new placeholders
        output_target = tf.placeholder(dtype, shape=[None, output_dim], name='targetOutput')

        # solvers
        with tf.name_scope('squareLoss'):
            core.square_loss_opt(dtype, output_target, output, tf.train.AdamOptimizer(learning_rate=self.learningRate))
