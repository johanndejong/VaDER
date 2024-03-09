import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
from scipy.stats import multivariate_normal
import warnings
from sklearn.mixture import GaussianMixture
import tensorflow_addons as tfa
import abc

from VaDER.utils import get_angles, positional_encoding, create_padding_mask, create_look_ahead_mask, \
    scaled_dot_product_attention, create_masks

class ImputationLayer(tf.keras.layers.Layer):
    def __init__(self, A_init):
        super(ImputationLayer, self).__init__()
        self.A = self.add_weight(
            "A", shape=A_init.shape, initializer=tf.constant_initializer(A_init))
    def call(self, X, W):
        return X * W + self.A * (1.0 - W)

class RnnDecodeTransformLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, I):
        super(RnnDecodeTransformLayer, self).__init__()
        weight_init = tf.constant_initializer(np.random.standard_normal([n_hidden, I]))
        bias_init = tf.constant_initializer(np.zeros(I) + 0.1)
        self.weight = self.add_weight(
            "weight", shape=[n_hidden, I], initializer=tf.initializers.glorot_uniform())
        self.bias = self.add_weight(
            "bias", shape=[I], initializer=tf.initializers.glorot_uniform())
    def call(self, rnn_output, batch_size):
        # rnn_output = tf.transpose(rnn_output, perm=[1, 0, 2])
        # rnn_output = tf.transpose(a=tf.stack(rnn_output), perm=[1, 0, 2])
        weight = tf.tile(tf.expand_dims(self.weight, 0), [batch_size, 1, 1])
        return tf.matmul(rnn_output, weight) + self.bias

class GmmLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, K):
        super(GmmLayer, self).__init__()
        self.mu_c_unscaled = self.add_weight(
            "mu_c_unscaled", shape=[K, n_hidden], initializer=tf.initializers.glorot_uniform())
        self.sigma2_c_unscaled = self.add_weight(
            "sigma2_c_unscaled", shape=[K, n_hidden], initializer=tf.initializers.glorot_uniform())
        self.phi_c_unscaled = self.add_weight(
            "phi_c_unscaled", shape=[K], initializer=tf.initializers.glorot_uniform())
    def call(self, _):
        mu_c = self.mu_c_unscaled
        sigma2_c = tf.nn.softplus(self.sigma2_c_unscaled, name="sigma2_c")
        phi_c = tf.nn.softmax(self.phi_c_unscaled, name="phi_c")
        return mu_c, sigma2_c, phi_c
