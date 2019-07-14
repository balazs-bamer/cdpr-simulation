import tensorflow as tf
import numpy as np

with tf.device('/cpu:0'):

    with tf.Session() as session:

        with session.graph.as_default():
            tf.set_random_seed(0)

        state_dim = 3
        action_dim = 1

        hidden_sizes = (400, 300)

        dtype = tf.float32

        update_batch_normalization_moving_averages = tf.cast(tf.reshape(tf.placeholder(dtype=dtype, shape=[1], name='updateBatchNormalizationMovingAverages'), shape=[]), dtype=tf.bool)

        state_input = tf.placeholder(dtype, shape=[None, state_dim], name='state')

        top = state_input

        with tf.name_scope('hiddenLayer1'):
            W1 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[state_dim, hidden_sizes[0]], minval=-np.sqrt(3.0/(state_dim)), maxval=np.sqrt(3.0/(state_dim))), name='W')
            b1 = tf.Variable(tf.constant(value=0.1, dtype=dtype, shape=[hidden_sizes[0]]), name='b')
            top = tf.matmul(top, W1) + b1
            top = tf.nn.relu(top)

        with tf.name_scope('hiddenLayer2'):
            W2 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[hidden_sizes[0], hidden_sizes[1]], minval=-np.sqrt(3.0/(hidden_sizes[0])), maxval=np.sqrt(3.0/(hidden_sizes[0]))), name='W')
            b2 = tf.Variable(tf.constant(value=0.1, dtype=dtype, shape=[hidden_sizes[1]]), name='b')
            top = tf.matmul(top, W2) + b2
            top = tf.nn.relu(top)

        with tf.name_scope('outputLayer'):
            Wo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[hidden_sizes[1], action_dim], minval=-3e-3, maxval=3e-3), name='W')
            bo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[action_dim], minval=-3e-3, maxval=3e-3), name='b')
            top = tf.matmul(top, Wo) + bo
            top = tf.tanh(top)

        action_output = tf.identity(top, name='action')

        parameters_to_be_updated_by_solver = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_parameters = [W1, b1, W2, b2, Wo, bo] + tf.get_collection('BATCH_NORM_VARIABLES')

        tf.add_check_numerics_ops()

        print "policy:"
        print "to be updated by solver: "
        for parameter in parameters_to_be_updated_by_solver:
            print parameter.name
        print "all_parameters: "
        for parameter in all_parameters:
            print parameter.name

        assign_op_list = []
        interpolate_op_list = []

        with tf.name_scope('learnableParameters'):
            tau = tf.placeholder(dtype, name='tau')
            tf.identity(tf.constant(value=len(all_parameters), dtype=tf.int32), name='numberOf')
            for idx, parameter in enumerate(all_parameters):
                with tf.device('/cpu:0'):
                    tf.identity(tf.constant(value=parameter.name, dtype=tf.string), name='name_%d'%idx)

                parameter_assign_placeholder = tf.placeholder(dtype, name='parameterPlaceholder_%d'%idx)
                assign_op_list += [parameter.assign(parameter_assign_placeholder)]
                interpolate_op_list += [parameter.assign(parameter_assign_placeholder*tau + parameter*(1-tau))]

            learnable_parameters_assign_all = tf.group(*assign_op_list, name='assignAll')
            learnable_parameters_interpolate_all = tf.group(*interpolate_op_list, name='interpolateAll')

        action_target = tf.placeholder(dtype, shape=[None, action_dim], name='targetAction')

        with tf.name_scope('trainUsingTargetAction'):
            loss = tf.reduce_mean(tf.square(action_output - action_target))

            regularization_term = tf.add_n([0*1e-7*tf.nn.l2_loss(var) for var in parameters_to_be_updated_by_solver])
            regularized_loss = loss + regularization_term

            gradient_of_loss_wrt_action_output = tf.gradients(loss, action_output)[0]

            train_using_action_target_learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            train_using_action_target_solver_op = tf.train.AdamOptimizer(learning_rate=train_using_action_target_learning_rate).minimize(regularized_loss, name='solver')

        with tf.name_scope('trainUsingCritic'):

            gradient_from_critic = tf.placeholder(dtype, shape=[None, action_dim], name='gradientFromCritic')

            train_using_critic_learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            train_using_critic_optimizer = tf.train.AdamOptimizer(learning_rate=train_using_critic_learning_rate)
            regularization_term = tf.add_n([0*1e-6*tf.nn.l2_loss(var) for var in parameters_to_be_updated_by_solver])
            manipulated_parameter_gradients = []
            for parameter in parameters_to_be_updated_by_solver:
                gradient_due_to_parameter_normalization = tf.gradients(regularization_term, parameter)[0]
                gradient_regularized = tf.gradients(action_output, parameter, gradient_from_critic)[0] + gradient_due_to_parameter_normalization
                manipulated_parameter_gradients += [gradient_regularized]
            manipulated_parameter_gradients_and_parameters = zip(manipulated_parameter_gradients, parameters_to_be_updated_by_solver)

            train_using_critic_apply_gradients = train_using_critic_optimizer.apply_gradients(manipulated_parameter_gradients_and_parameters, name='applyGradients')


        initialize_all_variables_op = tf.initialize_variables(tf.all_variables(), name='initializeAllVariables')

        summary_writer = tf.train.SummaryWriter('/tmp/debug_tf', graph_def=session.graph_def)

        tf.train.write_graph(session.graph_def, '', 'policy.pb', as_text=False)
