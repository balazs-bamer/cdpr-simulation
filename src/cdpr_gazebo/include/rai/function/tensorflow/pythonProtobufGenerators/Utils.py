import tensorflow as tf
import numpy as np


class Utils:
    def __init__(self, dtype):
        self.Dtype = dtype

    def var_shape(self, x):
        out = [k.value for k in x.get_shape()]
        assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
        return out

    def numel(self, x):
        return np.prod(self.var_shape(x))


    def flatgrad(self, loss, var_list, maxnorm=None):
        grads = tf.gradients(loss, var_list)
        if maxnorm is not None:
            grads, _ = tf.clip_by_global_norm(grads, maxnorm)
        return tf.reshape(tf.concat([tf.reshape(grad, [self.numel(v)]) for (v, grad) in zip(var_list, grads)], axis=0),
                          [1, -1])

    def log_likelihood(self, mean, stdev, sampledAction=None):
        dim = tf.cast(tf.shape(mean)[-1], dtype=self.Dtype)
        if sampledAction is None:
            return - 0.5 * tf.reduce_sum(tf.square(mean / stdev), axis=-1) - \
                   tf.reduce_sum(tf.log(stdev), axis=-1) - \
                   0.5 * tf.cast(tf.log(2.0 * np.pi), self.Dtype) * dim
        else:
            return - 0.5 * tf.reduce_sum(tf.square((sampledAction - mean) / stdev), axis=-1) - \
                   tf.reduce_sum(tf.log(stdev), axis=-1) - \
                   0.5 * tf.cast(tf.log(2.0 * np.pi), self.Dtype) * dim

    def kl_divergence(self, mean1, stdev1, mean2, stdev2):
        numerator = tf.square(mean1 - mean2) + tf.square(stdev1) - tf.square(stdev2)
        denominator = 2 * tf.square(stdev2) + 1e-8
        return tf.reduce_sum(tf.divide(numerator, denominator) + tf.log(stdev2 / stdev1), axis=-1)

    def CG_tf(self, eval, b, IterN=100, tol=1e-15, damping=0.1):
        # Not applicable to recurrent network(dynamic rnn)
        def cond(i, p, r, x, rdotr):
            return tf.logical_and(tf.less(tol, rdotr), tf.less(i, IterN))

        def iter(i, p, r, x, rdotr):
            Ap = eval(p)
            Ap += damping * p  # cg damping
            a = tf.divide(rdotr, tf.reduce_sum(tf.matmul(p, tf.transpose(Ap))))
            x += a * p
            r -= a * Ap
            newrdotr = tf.reduce_sum(tf.matmul(r, tf.transpose(r)))
            mu = newrdotr / rdotr
            p = r + mu * p
            return i + 1, p, r, x, newrdotr

        # Initialize
        r = tf.identity(b)
        p = tf.identity(b)
        x = tf.zeros(shape=tf.shape(b), dtype=self.Dtype)
        tol = tf.cast(tol, dtype=self.Dtype)
        initial_i = tf.constant(0)
        rdotr = tf.reduce_sum(tf.matmul(r, tf.transpose(r)))  # vector dot product

        i, p, r, x, rdotr = tf.while_loop(
            cond, iter,
            [initial_i, p, r, x, rdotr],
            shape_invariants=[initial_i.get_shape(), b.get_shape(), b.get_shape(), b.get_shape(), rdotr.get_shape()])

        return x, rdotr
