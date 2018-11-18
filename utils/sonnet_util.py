import tensorflow as tf
import sonnet as snt

from utils import tf_util as U


def parse_nonlin(nonlin_key):
    """Parse the activation function"""
    nonlin_map = dict(relu=tf.nn.relu,
                      leaky_relu=U.leaky_relu,
                      prelu=U.prelu,
                      elu=tf.nn.elu,
                      selu=U.selu,
                      tanh=tf.nn.tanh,
                      identity=tf.identity)
    if nonlin_key in nonlin_map.keys():
        return nonlin_map[nonlin_key]
    else:
        raise RuntimeError("unknown nonlinearity: '{}'".format(nonlin_key))


def parse_initializer(hid_w_init_key):
    """Parse the weight initializer"""
    init_map = dict(he_normal=U.he_normal_init(),
                    he_uniform=U.he_uniform_init(),
                    xavier_normal=U.xavier_normal_init(),
                    xavier_uniform=U.xavier_uniform_init())
    if hid_w_init_key in init_map.keys():
        return init_map[hid_w_init_key]
    else:
        raise RuntimeError("unknown weight init: '{}'".format(hid_w_init_key))


class PolicyNN(snt.AbstractModule):

    def __init__(self, scope, name, ac_space, hps):
        super(PolicyNN, self).__init__(name=name)
        self.scope = scope
        self.name = name
        self.ac_space = ac_space
        self.hps = hps

    def _build(self, ob):
        self.set_hid_initializers()
        self.set_out_initializers()
        self.set_hid_regularizer()
        embedding = ob
        embedding = self.stack_fc_layers(embedding)
        ac = self.add_out_layer(embedding)
        return ac

    def stack_fc_layers(self, embedding):
        """Stack the fully-connected layers
        Note that according to the paper 'Parameter Space Noise for Exploration', layer
        normalization should only be used for the fully-connected part of the network.
        """
        for hid_layer_index, hid_width in enumerate(self.hps.hid_widths, start=1):
            hid_layer_id = "fc{}".format(hid_layer_index)
            # Add hidden layer
            embedding = snt.Linear(output_size=hid_width, name=hid_layer_id,
                                   initializers=self.hid_initializers,
                                   regularizers=self.hid_regularizers)(embedding)
            # Add non-linearity
            embedding = parse_nonlin(self.hps.hid_nonlin)(embedding)
        return embedding

    def set_hid_initializers(self):
        self.hid_initializers = {'w': parse_initializer(self.hps.hid_w_init),
                                 'b': tf.zeros_initializer()}

    def set_out_initializers(self):
        self.out_initializers = {'w': tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                 'b': tf.zeros_initializer()}

    def set_hid_regularizer(self):
        self.hid_regularizers = {'w': U.weight_decay_regularizer(scale=0.)}

    def add_out_layer(self, embedding):
        """Add the output layer"""
        self.ac_dim = self.ac_space.shape[-1]  # num dims
        embedding = snt.Linear(output_size=self.ac_dim, name='final',
                               initializers=self.out_initializers)(embedding)
        return embedding

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope + "/" + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.scope + "/" + self.name)
