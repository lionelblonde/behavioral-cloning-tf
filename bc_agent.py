import tensorflow as tf

from gym import spaces

from utils import tf_util as U
import logger
from utils.misc_util import zipsame
from utils.console_util import columnize
from utils.sonnet_util import PolicyNN
from mpi_running_mean_std import MpiRunningMeanStd


class BCAgent(object):

    def __init__(self, name, env, hps):
        self.name = name
        # Define everything in a specific scope
        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name
            self._init(env=env, hps=hps)

    def _init(self, env, hps):
        # Parameters
        self.env = env
        self.ob_shape = self.env.observation_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape
        self.hps = hps

        self.policy_nn = PolicyNN(scope=self.scope, name='pol', ac_space=self.ac_space, hps=hps)

        # Create inputs
        e_obs = U.get_placeholder(name='e_obs', dtype=tf.float32, shape=(None,) + self.ob_shape)
        e_acs = U.get_placeholder(name='e_acs', dtype=tf.float32, shape=(None,) + self.ac_shape)

        # Rescale observations
        if self.hps.rmsify_obs:
            # Smooth out observations using running statistics and clip
            with tf.variable_scope("apply_obs_rms"):
                self.obs_rms = MpiRunningMeanStd(shape=self.ob_shape)
            e_obz = self.rmsify(e_obs, self.obs_rms)
        else:
            e_obz = e_obs

        # Build graphs
        self.ac_pred = self.policy_nn(e_obz)

        # Define loss
        if isinstance(self.ac_space, spaces.Box):
            self.loss = tf.reduce_mean(tf.square(e_acs - self.ac_pred))
        elif isinstance(self.ac_space, spaces.Discrete):
            # self.loss = tf.reduce_mean(tf.cast(tf.equal(e_acs, self.ac_pred), dtype=tf.float32))
            self.loss = tf.reduce_mean(tf.square(e_acs - self.ac_pred))
        else:
            raise RuntimeError("ac space is neither Box nor Discrete")

        # Create Theano-like ops
        self.act = U.function([e_obs], self.ac_pred)
        self.compute_pol_loss = U.function([e_obs, e_acs], self.loss)

        # Summarize module imformation in logs
        self.log_module_info(self.policy_nn)

    def predict(self, ob):
        """Predict an action from an observation"""
        ob_expanded = ob[None]
        ac_pred = self.act(ob_expanded)
        return ac_pred.flatten()

    def rmsify(self, x, x_rms):
        """Normalize `x` with running statistics"""
        assert x.dtype == tf.float32, "must be a tensor of the right dtype"
        rmsed_x = (x - x_rms.mean) / x_rms.std
        return rmsed_x

    def log_module_info(self, *components):
        assert len(components) > 0, "components list is empty"
        for component in components:
            logger.info("logging {}/{} specs".format(self.name, component.name))
            names = [var.name for var in component.trainable_vars]
            shapes = [U.var_shape(var) for var in component.trainable_vars]
            num_paramss = [U.numel(var) for var in component.trainable_vars]
            zipped_info = zipsame(names, shapes, num_paramss)
            logger.info(columnize(names=['name', 'shape', 'num_params'],
                                  tuples=zipped_info,
                                  widths=[36, 16, 10]))
            logger.info("  total num params: {}".format(sum(num_paramss)))

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
