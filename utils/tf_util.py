import copy
import os
import collections
import multiprocessing

import numpy as np
import tensorflow as tf

from utils.misc_util import zipsame
import logger


def function(inputs, outputs, updates=None, givens=None):
    """Port Theano's 'function' to TensorFlow.
    Take a bunch of TensorFlow placeholders and expressions computed based on those placeholders
    and produces f(inputs) -> outputs. Function f takes values to be fed to the input's
    placeholders and produces the values of the expressions in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Args:
        inputs (TensorFlow Tensor or Object with make_feed_dict): List of input tensors of type
            tf.placeholder, tf.constant or more generally any object with `make_feed_dict method`.
        outputs (TensorFlow Tensor): list of output tensors or a single output tensor of type(s)
            tf.Variable to be returned from the function.
            Returned value will also have the same shape.
        updates (list): List of TensorFlow operations
        givens (dict): Values already know for some or all of the input Tensors
    """
    if isinstance(outputs, list):
        return _TheanoFunction(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _TheanoFunction(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _TheanoFunction(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _TheanoFunction(object):
    """Theano's 'function' core"""

    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            fmtbool = type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0
            if not hasattr(inpt, 'make_feed_dict') and not fmtbool:
                assert 0, "inputs should all be phs, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args):
        assert len(args) <= len(self.inputs), "too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]

        return results


def switch(condition, then_expression, else_expression):
    """Switch between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression` should be symbolic
    tensors of the *same shape*.

    # Arguments
        condition: scalar tensor,
        then_expression: TensorFlow operation,
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)

    return x


def clip(clip_range):
    """Clip the variable values to remain within the clip range `(l_b, u_b)`
    `l_b` and `u_b` respectively depict the lower and upper bounds of the range,
    e.g.
        clip_obs = clip(observation_range)
        clipped_obs = clip_obs(obs)
    """
    def _clip(x):
        return tf.clip_by_value(x, min(clip_range), max(clip_range))
    return _clip


def reduce_var(x, axis=None, keepdims=False):
    """Build a Tensor for the variance of `x`"""
    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Build a Tensor for the standard deviation of `x`"""
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def leaky_relu(x, leak=0.2, name='leaky_relu'):
    """Leaky ReLU activation function
    'Rectifier Nonlinearities Improve Neural Network Acoustic Models'
    AL Maas, AY Hannun, AY Ng, ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    Alternate implementation that might be more efficient:
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    [relies on max(0, x) = (x + abs(x))/2]
    """
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def prelu(x, name='prelu'):
    """Parametric ReLU activation function
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    He Kaiming, ICCV 2015, http://arxiv.org/abs/1502.01852
    """
    with tf.variable_scope(name):
        leak = tf.get_variable('leak', x.get_shape()[-1], initializer=tf.zeros_initializer())
        return tf.maximum(x, leak * x)


def selu(x, name='selu'):
    """Scaled ELU activation function
    'Self-Normalizing Neural Networks'
    Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter, NIPS 2017,
    https://arxiv.org/abs/1706.02515
    Correct alpha and scale values for unit activation with mean zero and unit variance
    from https://github.com/bioinf-jku/SNNs/blob/master/selu.py#L24:9
    """
    with tf.variable_scope(name):
        # Values of alpha and scale corresponding to zero mean and unit variance
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # The authors' implementation returns
        #     scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
        # We replace tf.nn.elu by its expression
        #     tf.exp(x) - 1.     for x < 0
        #     x                  for x >= 0
        # By substitution, scale * tf.where(x >= 0., x, alpha * (tf.exp(x) - 1.))
        return scale * tf.where(x >= 0., x, alpha * tf.exp(x) - alpha)


def huber_loss(x, delta=1.0, name='huber_loss'):
    """Less sensitive to outliers than the l2 loss while being differentiable at 0
    Reference: https://en.wikipedia.org/wiki/Huber_loss
    """
    with tf.variable_scope(name):
        return tf.where(tf.abs(x) < delta,
                        0.5 * tf.square(x),
                        delta * (tf.abs(x) - 0.5 * delta))


def logit_bernoulli_entropy(logits):
    """Entropy of a Bernoulli distribution from logits
    Note that the log-sigmoid function is defined by y = log(1 / 1 + exp(-x))
    However, for numerical stability, tf.log_sigmoid is implemented as
    y = -tf.nn.softplus(-x)
    How to find the formula: write the Bernoulli entropy with p = sig(logits)
    """
    ent = (1. - tf.sigmoid(logits)) * logits - tf.log_sigmoid(logits)

    return ent


def batch_size(x):
    """Returns an int corresponding to the batch size of the input tensor"""
    return tf.to_float(tf.shape(x)[0], name='get_batch_size_in_fl32')


def make_session(tf_config=None, num_core=None, make_default=False, graph=None):
    """Returns a session which will use <num_threads> threads only
    It does not always ensure better performance: https://stackoverflow.com/a/39395608
    Prefer MPI parallelism over tf
    """
    assert isinstance(tf_config, tf.ConfigProto) or tf_config is None
    if num_core is None:
        # Num of cores can also be communicated from the exterior, via an env var
        num_core = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if tf_config is None:
        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_core,
            intra_op_parallelism_threads=num_core)
        tf_config.gpu_options.allow_growth = True
    if make_default:
        # The only difference with a regular Session is that
        # an InteractiveSession installs itself as the default session on construction
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def get_available_gpus():
    """Return the current available GPUs via tensorflow API
    From stackoverflow post:
        https://stackoverflow.com/questions/
        38559755/how-to-get-current-available-gpus-in-tensorflow?
        utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def single_threaded_session():
    """Returns a session which will only use a single core"""
    return make_session(num_core=1)


ALREADY_INITIALIZED = set()  # python set: unordered collection unique elements


def initialize():
    """Initialize all the uninitialized variables in the global scope"""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    tf.get_default_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def file_writer(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return tf.summary.FileWriter(dir_path, tf.get_default_session().graph)


def load_model(model, var_list=None):
    """Restore variables from disk
    `model` is the path of the model checkpoint to load
    """
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(tf.get_default_session(), model)


def load_latest_checkpoint(model_dir, var_list=None):
    """Restore variables from disk
    `model_dir` is the path of the directory containing all the checkpoints
    for a given experiment
    """
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(tf.get_default_session(), tf.train.latest_checkpoint(model_dir))


def save_state(fname, var_list=None, iters_so_far=None):
    """Save the variables to disk"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # No exception is raised if the directory already exists
    saver = tf.train.Saver(var_list=var_list)
    saver.save(tf.get_default_session(), fname, global_step=iters_so_far)


def extract_fan_in_out(shape):
    fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
    fan_out = float(shape[-1])
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)
    return fan_in, fan_out


def xavier_uniform_init():
    """Xavier uniform initialization
    'Understanding the difficulty of training deep feedforward neural networks'
    Xavier Glorot & Yoshua Bengio, AISTATS 2010,
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Draws samples from a uniform distribution within [-w_bound, w_bound],
    with w_bound = np.sqrt(6.0 / (fan_in + fan_out))
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, fan_out = extract_fan_in_out(shape)
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(shape=shape, minval=-w_bound, maxval=w_bound, dtype=dtype)
    return _initializer


def xavier_normal_init():
    """Xavier normal initialization
    'Understanding the difficulty of training deep feedforward neural networks'
    Xavier Glorot & Yoshua Bengio, AISTATS 2010,
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Draws samples from a truncated normal distribution centered on 0,
    with stddev = np.sqrt(2.0 / (fan_in + fan_out)) i.e. np.sqrt(1.0 / fan_avg)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, fan_out = extract_fan_in_out(shape)
        # Without truncation: stddev = np.sqrt(2.0 / (fan_in + fan_out))
        trunc_stddev = np.sqrt(1.3 * 2.0 / (fan_in + fan_out))
        return tf.truncated_normal(shape=shape, mean=0.0, stddev=trunc_stddev, dtype=dtype)
    return _initializer


def he_uniform_init():
    """He (MSRA) uniform initialization
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    Kaiming He, ICCV 2015, http://arxiv.org/abs/1502.01852
    Xavier initialization is justified when only linear activations are used,
    He initialization is justified when ReLU/leaky RELU/PReLU activations are used.

    Draws samples from a uniform distribution within [-w_bound, w_bound],
    with w_bound = np.sqrt(6.0 / fan_in)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, _ = extract_fan_in_out(shape)
        w_bound = np.sqrt(6.0 / fan_in)
        return tf.random_uniform(shape=shape, minval=-w_bound, maxval=w_bound, dtype=dtype)
    return _initializer


def he_normal_init():
    """He (MSRA) normal initialization
    'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification'
    He Kaiming, ICCV 2015, http://arxiv.org/abs/1502.01852
    Xavier initialization is justified when only linear activations are used,
    He initialization is justified when ReLU/leaky RELU/PReLU activations are used.

    Draws samples from a truncated normal distribution centered on 0,
    with stddev = np.sqrt(2.0 / fan_in)
    """
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in, _ = extract_fan_in_out(shape)
        # Without truncation: stddev = np.sqrt(2.0 / fan_in)
        trunc_stddev = np.sqrt(1.3 * 2.0 / fan_in)
        return tf.truncated_normal(shape=shape, mean=0.0, stddev=trunc_stddev, dtype=dtype)
    return _initializer


def weight_decay_regularizer(scale):
    """Apply l2 regularization on weights"""
    assert scale >= 0
    if scale == 0:
        return lambda _: None

    def _regularizer(weights):
        scale_tensor = tf.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
        return tf.multiply(scale_tensor, tf.nn.l2_loss(weights), name='weight_decay_loss')

    return _regularizer


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    """Returns the number of elements in `x`"""
    return intprod(var_shape(x))


def flatgrad(loss, var_list, clip_norm=None):
    """Returns a list of sum(dy/dx) for each x in `var_list`
    Clipping is done by global norm (paper: https://arxiv.org/abs/1211.5063)
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    vars_and_grads = zipsame(var_list, grads)  # zip with extra security
    for index, (var, grad) in enumerate(vars_and_grads):
        # If the gradient gets stopped for some obsure reason, set the grad as zero vector
        _grad = grad if grad is not None else tf.zeros_like(var)
        # Reshape the grad into a vector
        grads[index] = tf.reshape(_grad, [numel(var)])
    # return tf.concat(grads, axis=0)
    return tf.concat(grads, axis=0)


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)  # creates an op that groups multiple operations

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)


_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)


def print_ph_cache():
    """Print out the placeholders stored in the placeholder cache"""
    logger.info("Num of phs in cache: {}".format(len(_PLACEHOLDER_CACHE)))
    for k, v in _PLACEHOLDER_CACHE.items():
        logger.info("  key: {}".format(k))


def get_placeholder(name, dtype, shape):
    """Return a placeholder if already available in the current graph
    with the right shape. Otherwise, create the placeholder with the
    desired shape and return it.
    """
    if name in _PLACEHOLDER_CACHE:
        placeholder_, dtype_, shape_ = _PLACEHOLDER_CACHE[name]
        if placeholder_.graph == tf.get_default_graph():
            assert dtype_ == dtype and shape_ == shape, \
                "Placeholder with name {} has already been registered and has shape {}, \
                 different from requested {}".format(name, shape_, shape)
            return placeholder_
    placeholder_ = tf.placeholder(dtype=dtype, shape=shape, name=name)
    _PLACEHOLDER_CACHE[name] = (placeholder_, dtype, shape)
    return placeholder_


def get_placeholder_cached(name):
    """Returns an error if the placeholder does not exist"""
    return _PLACEHOLDER_CACHE[name][0]

