import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
from scipy.stats import multivariate_normal
import warnings
from sklearn.mixture import GaussianMixture
import tensorflow_addons as tfa
import abc

from VaDER.layers import ImputationLayer, RnnDecodeTransformLayer, GmmLayer, MultiHeadAttention, \
    point_wise_feed_forward_network, TransformerEncoderLayer, TransformerDecoderLayer, \
    TransformerEncoder, TransformerDecoder
from VaDER.utils import get_angles, positional_encoding, create_padding_mask, create_look_ahead_mask, \
    scaled_dot_product_attention, create_masks


class VaderModel(tf.keras.Model):
    def __init__(self, X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation, cell_params=None):
        super(VaderModel, self).__init__()
        self.D = D
        self.K = K
        self.I = I
        self.cell_type = cell_type
        self.n_hidden = n_hidden
        self.recurrent = recurrent
        self.output_activation = output_activation
        def initialize_imputation(X, W):
            # average per time point, variable
            W_A = np.sum(W, axis=0)
            A = np.sum(X * W, axis=0)
            A[W_A > 0] = A[W_A > 0] / W_A[W_A > 0]
            # if not available, then average across entire variable
            if recurrent:
                for i in np.arange(A.shape[0]):
                    for j in np.arange(A.shape[1]):
                        if W_A[i, j] == 0:
                            A[i, j] = np.sum(X[:, :, j]) / np.sum(W[:, :, j])
                            W_A[i, j] = 1
            # if not available, then average across all variables
            A[W_A == 0] = np.mean(X[W == 1])
            return A.astype(X.dtype)

        def sample(params):
            mu_tilde = params[0]
            log_sigma2_tilde = params[1]
            noise = tf.random.normal(tf.shape(input=log_sigma2_tilde))
            return tf.add(mu_tilde, tf.exp(0.5 * log_sigma2_tilde) * noise, name="z")

        self.imputation_layer = ImputationLayer(initialize_imputation(X, W))
        self.gmm_layer = GmmLayer(n_hidden[-1], K)
        self.z_layer = tf.keras.layers.Lambda(sample, name="z")

    @tf.function
    @abc.abstractmethod
    def encode(self, X, W, training=False):
        pass

    @tf.function(experimental_relax_shapes=True)
    @abc.abstractmethod
    def decode(self, z, X=None, training=False):
        pass

    @tf.function
    @abc.abstractmethod
    def call(self, inputs, training=False):
        pass

class VaderRNN(VaderModel):
    def __init__(self, X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation, cell_params=None):
        super(VaderRNN, self).__init__(X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation)
        if len(n_hidden) > 1:
            if cell_type == "LSTM":
                encoder = tfa.rnn.PeepholeLSTMCell(n_hidden[0], activation=tf.nn.tanh, name="encoder")
                decoder = tfa.rnn.PeepholeLSTMCell(n_hidden[0], activation=tf.nn.tanh, name="decoder")
            elif cell_type == "GRU":
                encoder = tf.keras.layers.GRUCell(n_hidden[0], activation=tf.nn.tanh, name="encoder")
                decoder = tf.keras.layers.GRUCell(n_hidden[0], activation=tf.nn.tanh, name="decoder")
            else:
                encoder = tf.keras.layers.SimpleRNNCell(n_hidden[0], activation=tf.nn.tanh, name="encoder")
                decoder = tf.keras.layers.SimpleRNNCell(n_hidden[0], activation=tf.nn.tanh, name="decoder")
            self.ae_encode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in n_hidden[1:-1]]
            self.mu_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="mu_tilde")
            self.log_sigma2_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="log_sigma2_tilde")
            self.rnn_transform_layer = RnnDecodeTransformLayer(n_hidden[0], I)
            self.ae_decode_layers = [
                tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in (n_hidden[:-1])[::-1]]
        else:
            n_hidden = n_hidden[0]
            if cell_type == "LSTM":
                # one n_hidden for mu, one for sigma2
                encoder = tf.keras.experimental.PeepholeLSTMCell(
                    n_hidden + n_hidden, activation=tf.nn.tanh, name="encoder")
                decoder = tf.keras.experimental.PeepholeLSTMCell(n_hidden, activation=tf.nn.tanh, name="decoder")
            elif cell_type == "GRU":
                # one n_hidden for mu, one for sigma2
                encoder = tf.keras.layers.GRUCell(n_hidden + n_hidden, activation=tf.nn.tanh, name="encoder")
                decoder = tf.keras.layers.GRUCell(n_hidden, activation=tf.nn.tanh, name="decoder")
            else:
                encoder = tf.keras.layers.SimpleRNNCell(n_hidden + n_hidden, activation=tf.nn.tanh, name="encoder")
                decoder = tf.keras.layers.SimpleRNNCell(n_hidden, activation=tf.nn.tanh, name="decoder")
            self.mu_layer = tf.keras.layers.Lambda(lambda hidden: hidden[:, :n_hidden], name="mu_tilde")
            self.log_sigma2_layer = tf.keras.layers.Lambda(
                lambda hidden: hidden[:, n_hidden:], name="log_sigma2_tilde")
            self.rnn_transform_layer = RnnDecodeTransformLayer(n_hidden, I)
        # select the carry state (not the memory state)
        # unroll = True raises "ValueError: Cannot unroll a RNN if the time dimension is undefined."
        self.encoder_rnn = tf.keras.layers.RNN(encoder, unroll=False, return_state=True, return_sequences=True)
        # select the output
        self.decoder_rnn = tf.keras.layers.RNN(decoder, unroll=False, return_state=True, return_sequences=True)

    @tf.function
    def encode(self, X, W, training=False):
        X_imputed = self.imputation_layer(X, W)
        # X_imputed = [tf.squeeze(t, [1]) for t in tf.split(X_imputed, self.D, 1)]
        hidden = self.encoder_rnn(X_imputed)[-1]  # [1] ???
        if len(self.n_hidden) > 1:
            for layer in self.ae_encode_layers:
                hidden = layer(hidden)
        mu_tilde = self.mu_layer(hidden)
        log_sigma2_tilde = self.log_sigma2_layer(hidden)
        z = self.z_layer((mu_tilde, log_sigma2_tilde))
        return z, mu_tilde, log_sigma2_tilde

    @tf.function(experimental_relax_shapes=True)
    def decode(self, z, X=None, training=False):
        hidden = z
        if len(self.n_hidden) > 1:
            for layer in self.ae_decode_layers:
                hidden = layer(hidden)
        inputs = tf.zeros((tf.shape(z)[0], self.D, self.I))
        if self.cell_type == "LSTM":
            initial_state = (tf.zeros(tf.shape(input=hidden)), hidden)
        else:
            initial_state = hidden
        rnn_output = self.decoder_rnn(inputs=inputs, initial_state=initial_state)[0]
        x_raw = self.rnn_transform_layer(rnn_output, tf.shape(z)[0])
        x = self.output_activation(x_raw, name="x_output")
        return x, x_raw

    @tf.function
    def call(self, inputs, training=False):
        X = inputs[0]
        W = inputs[1]
        z, mu_tilde, log_sigma2_tilde = self.encode(X, W)
        x, x_raw = self.decode(z)
        dummy_val = tf.constant(0.0, dtype=x.dtype)
        mu_c, sigma2_c, phi_c = self.gmm_layer(dummy_val)
        return x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde

class VaderFFN(VaderModel):
    def __init__(self, X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation, cell_params=None):
        super(VaderFFN, self).__init__(X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation)
        self.ae_encode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in n_hidden[:-1]]
        self.mu_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="mu_tilde")
        self.log_sigma2_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="log_sigma2_tilde")
        self.ae_decode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in (n_hidden[:-1])[::-1]]
        self.x_raw_layer = tf.keras.layers.Dense(D, activation=None, name="x_raw")

    @tf.function
    def encode(self, X, W, training=False):
        X_imputed = self.imputation_layer(X, W)
        hidden = X_imputed
        if len(self.n_hidden) > 1:
            for layer in self.ae_encode_layers:
                hidden = layer(hidden)
        mu_tilde = self.mu_layer(hidden)
        log_sigma2_tilde = self.log_sigma2_layer(hidden)
        z = self.z_layer((mu_tilde, log_sigma2_tilde))
        return z, mu_tilde, log_sigma2_tilde

    @tf.function(experimental_relax_shapes=True)
    def decode(self, z, X=None, training=False):
        hidden = z
        if len(self.n_hidden) > 1:
            for layer in self.ae_decode_layers:
                hidden = layer(hidden)
        x_raw = self.x_raw_layer(hidden)
        x = self.output_activation(x_raw, name="x_output")
        return x, x_raw

    @tf.function
    def call(self, inputs, training=False):
        X = inputs[0]
        W = inputs[1]
        z, mu_tilde, log_sigma2_tilde = self.encode(X, W)
        x, x_raw = self.decode(z)
        dummy_val = tf.constant(0.0, dtype=x.dtype)
        mu_c, sigma2_c, phi_c = self.gmm_layer(dummy_val)
        return x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde
