import tensorflow as tf
import numpy as np

import logger


class CustomSummary():

    def __init__(self, scalar_keys=[], histogram_keys=[], family=None):
        self.scalar_keys = scalar_keys
        self.histogram_keys = histogram_keys
        self.scalar_summaries = []
        self.scalar_summaries_ph = []
        self.histogram_summaries_ph = []
        self.histogram_summaries = []
        self.family = family
        with tf.variable_scope('summary'):
            for k in scalar_keys:
                ph = tf.placeholder('float32', None, name=k + '.scalar.summary')
                sm = tf.summary.scalar(k + '.scalar.summary', ph, family=self.family)
                self.scalar_summaries_ph.append(ph)
                self.scalar_summaries.append(sm)
            for k in histogram_keys:
                ph = tf.placeholder('float32', None, name=k + '.histogram.summary')
                sm = tf.summary.scalar(k + '.histogram.summary', ph, family=self.family)
                self.histogram_summaries_ph.append(ph)
                self.histogram_summaries.append(sm)

        # Creates a `Summary` protocol buffer that contains the union of all the values
        # in the input summaries
        self.summaries = tf.summary.merge(self.scalar_summaries + self.histogram_summaries)

    def add_all_summaries(self, writer, values, iteration):
        """Populate the summaries with a value for each key specified at instantiation.
        Note that the order of the incoming `values` should be the same as the that of the
        `scalar_keys` given in `__init__`"""
        if np.sum(np.isnan(values) + 0) != 0:  # '+0' to transform the bool into int
            # Return None if there is a NaN in the values
            logger.info("SUMMARIES CORRUPTED -> UNREADABLE IN TENSORBOARD")
            logger.info("  number of NaNs: {}".format(np.sum(np.isnan(values))))
            return
        sess = tf.get_default_session()
        keys = self.scalar_summaries_ph + self.histogram_summaries_ph
        # Build and populate the dict of key-value pairs to be fed to the `self.summaries` op
        feed_dict = {}
        for k, v in zip(keys, values):
            feed_dict.update({k: v})
        # Feed the dict to the summary operator
        summaries_str = sess.run(self.summaries, feed_dict)
        # Wrap the provided summary in an `Event` protocol buffer and adds it to the event file
        # `add_summary` can pass the result of evaluating any summary op (using e.g. `sess.run`)
        writer.add_summary(summaries_str, iteration)
        writer.flush()
