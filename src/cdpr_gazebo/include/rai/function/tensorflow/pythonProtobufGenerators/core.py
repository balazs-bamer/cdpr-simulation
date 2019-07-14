import tensorflow as tf
# from tensorflow.contrib.keras import backend as K
# import tensorflow.contrib.keras as keras
# from tensorflow.contrib.keras import layers
# from tensorflow.contrib.keras import models
from tensorflow.python.client import device_lib
import os


# Device configure
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def dev_config(dev_info):
    command = dev_info.split(',')
    gpulist = get_available_gpus()
    gpuset = set(gpulist)
    opt = len(command)
    dev_list = []
    dev_mode = False

    if command[0] == 'cpu':
        dev_list = ['/cpu:0']
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    elif command[0] == 'gpu':
        assert opt > 1, 'Specify at least one gpu (e.g. gpu,0 or gpu,0,1)'
        dev_mode = True
        temp = []
        for i in range(1, opt):
            dev = '/gpu:' + command[i]
            dev_list.append(dev)
        #     if dev in gpuset:  # check availability
        #         dev_list.append(dev)
        #     else:
        #         temp.append(dev)
        # if (opt - 1) != len(dev_list):
        #     print('Cannot find :' + temp)
    assert len(dev_list) > 0, 'Cannot find any device'
    return dev_mode, dev_list


def gpu_config(dev_list, gs):
    print('Selected devices:')
    print(dev_list)
    # GPU configuration(Data parallelism)
    if gs.net is not None:
        if len(dev_list) > 1:
            gs.net = gpu_parallel(gs.net, dev_list)
        gs.output = gs.net.output
    else:
        print('This graph structure does not support multiple GPUs')

#
# def gpu_parallel(model, gpu_list):
#     def get_slice(data, idx, parts):  # TODO : fix
#         shape = tf.shape(data)
#         stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
#         start = stride * (parts - idx)  # backward
#         if idx == 1:  # assign remaining data to the first device
#             size = tf.concat([shape[:1] // parts + shape[:1] % parts, shape[1:]], axis=0)
#         else:
#             size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
#         return tf.slice(data, start, size)
#
#     outputs_all = []
#     for i in range(len(model.outputs)):
#         outputs_all.append([])
#
#     gpu_count = len(gpu_list)
#     cnt = 1
#     # Place a copy of the model on each GPU, each getting a slice of the batch
#     for d in gpu_list:
#         with tf.device(d):
#             with tf.name_scope('tower_%d' % cnt) as scope:
#
#                 inputs = []
#                 # Slice each input into a piece for processing on this GPU
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_n = layers.Lambda(get_slice,
#                                             arguments={'idx': cnt, 'parts': gpu_count})(x)
#                     inputs.append(slice_n)
#                 outputs = model(inputs)
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]
#                 # Save all the outputs for merging back together later
#                 for l in range(len(outputs)):
#                     outputs_all[l].append(outputs[l])
#         cnt = cnt + 1
#
#     # merge outputs
#     # with tf.device('/cpu:0'):
#     with tf.device(gpu_list[0]):
#         merged = []
#         if len(outputs_all[0]) == 1:
#             merged = outputs_all[0]
#         else:
#             for outputs in outputs_all:
#                 merged.append(layers.concatenate(inputs=outputs, axis=0))
#     return models.Model(inputs=model.inputs, outputs=merged)

# Keras
# class Paramlayer(layers.Layer):
#     def __init__(self, dtype, **kwargs):
#         self.Dtype = tf.as_dtype(dtype).as_numpy_dtype
#         super(Paramlayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.Param = self.add_weight(shape=[1, input_shape[1]], initializer=keras.initializers.Zeros,
#                                      trainable=True, name='Extraparam', dtype=self.Dtype)
#         super(Paramlayer, self).build(input_shape)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1])
#
#     def call(self, x, **kwargs):
#         return self.Param

# loss function related
def huber_loss(labels, predictions, delta=1e-1, extraCost=0, name=None):
    cutoffline = tf.constant(value=delta, dtype=labels.dtype)
    residual = tf.sqrt(tf.reduce_sum(tf.square(predictions - labels), axis=1))
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = cutoffline * residual - 0.5 * tf.square(cutoffline)
    return tf.reduce_mean(tf.where(condition, small_res, large_res), 0, name=name) + extraCost


def infimum_loss(labels, predictions, underesterror=1e-3, extraCost=0, name=None):
    residual = labels - predictions
    condition = tf.less(residual, 0)
    small_res = 0.5 * tf.square(residual)
    large_res = underesterror * residual
    return tf.reduce_mean(tf.where(condition, small_res, large_res), 0, name=name) + extraCost

def huber_loss_opt(dtype, target, value, optimizer, extraCost=0, maxnorm = None):
    loss = huber_loss(target, value, extraCost, name='loss')
    grad = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    if maxnorm is not None:
        grads, variables = zip(*grad)
        grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
        grad = zip(grads, variables)
    train = optimizer.apply_gradients(grad, name='solver', global_step=tf.train.get_global_step())

def infimum_loss_opt(dtype, target, value, optimizer, extraCost=0, maxnorm = None):
    loss = infimum_loss(target, value, extraCost, name='loss')
    grad = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    if maxnorm is not None:
        grads, variables = zip(*grad)
        grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
        grad = zip(grads, variables)
    train = optimizer.apply_gradients(grad, name='solver', global_step=tf.train.get_global_step())

def square_loss_opt(dtype, target, value, optimizer, extraCost=0, maxnorm = None):
    loss = tf.reduce_mean(tf.square(target - value), name='loss') + extraCost
    grad = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    if maxnorm is not None:
        grads, variables = zip(*grad)
        grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
        grad = zip(grads, variables)
    train = optimizer.apply_gradients(grad, name='solver', global_step=tf.train.get_global_step())

def square_loss_trust_region_opt(dtype, target, value, value_prev, optimizer, clipRange = 0.2, extraCost = 0, maxnorm = None):
    # Clipping-based trust region loss (https://github.com/openai/baselines/blob/master/baselines/pposgd/pposgd_simple.py)
    clipped_value = value_prev + tf.clip_by_value(value - value_prev , -clipRange, clipRange)
    loss1 = tf.square(value - target)
    loss2 = tf.square(clipped_value - target)
    TR_loss = .5 * tf.reduce_mean(tf.maximum(loss1, loss2), name='loss') + extraCost
    grad = optimizer.compute_gradients(TR_loss, colocate_gradients_with_ops=True)
    if maxnorm is not None:
        grads, variables = zip(*grad)
        grads, gradnorm = tf.clip_by_global_norm(grads, clip_norm=maxnorm)
        grad = zip(grads, variables)
    train = optimizer.apply_gradients(grad, name='solver', global_step=tf.train.get_global_step())
