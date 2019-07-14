import functions.Policy as pc
import tensorflow as tf
import core


class DeterministicPolicy(pc.Policy):
    def __init__(self, dtype, gs):
        super(DeterministicPolicy, self).__init__(dtype, gs)

        # shortcuts
        action_dim = int(gs.output.shape[1])
        action = tf.identity(gs.output, name=self.output_names[0])
        state = gs.input

        # new placeholders
        action_target = tf.placeholder(dtype, shape=[None, action_dim], name='targetAction')

        # gradients
        jac_Action_wrt_Param = tf.concat([tf.reshape(tf.concat([tf.reshape(tf.gradients(action[:, idx], param), [-1]) for param in gs.l_param_list], axis=0), [-1, 1])
                                          for idx in range(action_dim)], name='jac_Action_wrt_Param', axis=1)
        jac_Action_wrt_State = tf.identity(tf.stack([tf.gradients(action[:, idx], state) for idx in range(action_dim)], axis=3)[0, 0, :, :], name='jac_Action_wrt_State')

        with tf.name_scope('trainUsingCritic'):
            gradient_placeholder = tf.placeholder(dtype, name='gradientFromCritic')
            gradient_from_critic = tf.reshape(gradient_placeholder, shape=[-1, action_dim])
            train_using_critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            grads = tf.gradients(action, gs.l_param_list, gradient_from_critic)
            grad_norm = tf.reduce_sum([tf.norm(grad) for grad in grads], name='gradnorm')
            manipulated_parameter_gradients_and_parameters = zip(grads, gs.l_param_list)
            train_using_critic_apply_gradients = train_using_critic_optimizer.apply_gradients(manipulated_parameter_gradients_and_parameters, name='applyGradients')
