import BaseClasses as bc
import tensorflow as tf
import core


class DeterministicRecurrentModel(bc.SpecializedFunction):

    input_names = ['state', 'action']
    output_names = ['nextState', 'hiddenState']

    def __init__(self, dtype, gs):
        super(DeterministicRecurrentModel, self).__init__(dtype, gs)

        # shortcuts
        state_dim = int(gs.input1.shape[0])
        action_dim = int(gs.input2.shape[1])
        hiddenState_dim = int(gs.output.shape[1])

        state = tf.identity(gs.input1, name=self.input_names[0])
        action = tf.identity(gs.input2, name=self.input_names[1])
        hiddenState = tf.identity(gs.output2, name=self.output_names[1])
        nextState = tf.identity(gs.output1, name=self.input_names[0])

        # new placeholders
        nextState_target = tf.placeholder(dtype, shape=[None, state_dim], name='nextState_target')

        # solvers
        with tf.name_scope('trainUsingTargetQValue'):
            core.square_loss_opt(dtype, nextState_target, nextState, tf.train.AdamOptimizer(learning_rate=self.learningRate))

        with tf.name_scope('trainUsingTargetQValue_huber'):
            core.huber_loss_opt(dtype, nextState_target, nextState, tf.train.AdamOptimizer(learning_rate=self.learningRate))
