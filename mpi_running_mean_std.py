from mpi4py import MPI

import numpy as np
import tensorflow as tf

from utils import tf_util as U


class MpiRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) -
                                      tf.square(self.mean),
                                      1e-2))

        newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        updates = [tf.assign_add(self._sum, newsum),
                   tf.assign_add(self._sumsq, newsumsq),
                   tf.assign_add(self._count, newcount)]
        self.incfiltparams = U.function([newsum, newsumsq, newcount], [], updates=updates)

    def update(self, x, comm):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(),
                                 np.square(x).sum(axis=0).ravel(),
                                 np.array([len(x)], dtype='float64')])
        comm.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape),
                           totalvec[n:2 * n].reshape(self.shape),
                           totalvec[2 * n])
