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
        action_input = tf.placeholder(dtype, shape=[None, action_dim], name='action')

        top = state_input

        with tf.name_scope('hiddenLayer1'):
            W1 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[state_dim, hidden_sizes[0]], minval=-np.sqrt(3.0/(state_dim)), maxval=np.sqrt(3.0/(state_dim))), name='W')
            b1 = tf.Variable(tf.constant(value=0.1, dtype=dtype, shape=[hidden_sizes[0]]), name='b')
            top = tf.matmul(top, W1) + b1
            top = tf.nn.relu(top)

        with tf.name_scope('hiddenLayer2'):
            top = tf.concat(1, [top, action_input])
            W2 = tf.Variable(tf.random_uniform(dtype=dtype, shape=[hidden_sizes[0]+action_dim, hidden_sizes[1]], minval=-np.sqrt(3.0/(hidden_sizes[0]+action_dim)), maxval=np.sqrt(3.0/(hidden_sizes[0]+action_dim))), name='W')
            b2 = tf.Variable(tf.constant(value=0.1, dtype=dtype, shape=[hidden_sizes[1]]), name='b')
            top = tf.matmul(top, W2) + b2
            top = tf.nn.relu(top)

        with tf.name_scope('outputLayer'):
            Wo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[hidden_sizes[1], 1], minval=-3e-3, maxval=3e-3), name='W')
            bo = tf.Variable(tf.random_uniform(dtype=dtype, shape=[1], minval=-3e-3, maxval=3e-3), name='b')
            top = tf.matmul(top, Wo) + bo

        q_value_output = tf.identity(top, name='QValue')

        parameters_to_be_updated_by_solver = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_parameters = [W1, b1, W2, b2, Wo, bo] + tf.get_collection('BATCH_NORM_VARIABLES')

        tf.add_check_numerics_ops()

        print "Qfunction: "
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

        q_value_target = tf.placeholder(dtype, shape=[None, 1], name='targetQValue')

        with tf.name_scope('trainUsingTargetQValue'):
            q_value_target_loss = tf.nn.l2_loss(q_value_output - q_value_target)

            q_value_target_loss_regularization_term = tf.add_n([0*1e-4*tf.nn.l2_loss(var) for var in parameters_to_be_updated_by_solver])
            q_value_target_loss_regularized_loss = q_value_target_loss + q_value_target_loss_regularization_term

            train_using_q_value_target_learning_rate = tf.reshape(tf.placeholder(dtype, shape=[1], name='learningRate'), shape=[])
            train_using_q_value_target = tf.train.AdamOptimizer(learning_rate=train_using_q_value_target_learning_rate).minimize(q_value_target_loss_regularized_loss, name='solver')

        gradient_of_sum_over_q_values_wrt_action = tf.identity(tf.gradients(tf.reduce_mean(q_value_output), action_input)[0], name='gradientOfSumOverQValuesWrtAction')


        initialize_all_variables_op = tf.initialize_variables(tf.all_variables(), name='initializeAllVariables')


        summary_writer = tf.train.SummaryWriter('/tmp/debug_tf', graph_def=session.graph_def)

        tf.train.write_graph(session.graph_def, '', 'Qfunction.pb', as_text=False)
