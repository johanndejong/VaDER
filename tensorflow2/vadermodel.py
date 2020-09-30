import tensorflow as tf
import numpy as np

class ImputationLayer(tf.keras.layers.Layer):
    def __init__(self, A_init):
        super(ImputationLayer, self).__init__()
        self.A = self.add_weight(
            "A", shape=A_init.shape, initializer=tf.constant_initializer(A_init))
    def call(self, X, W):
        W = tf.cast(W, X.dtype)
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

class VaderModel(tf.keras.Model):
    def __init__(self, X, W, D, K, I, cell_type, n_hidden, recurrent, output_activation):
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
            return A.astype(np.float32)

        def sample(params):
            mu_tilde = params[0]
            log_sigma2_tilde = params[1]
            noise = tf.random.normal(tf.shape(input=log_sigma2_tilde))
            return tf.add(mu_tilde, tf.exp(0.5 * log_sigma2_tilde) * noise, name="z")

        self.imputation_layer = ImputationLayer(initialize_imputation(X, W))
        self.gmm_layer = GmmLayer(n_hidden[-1], K)
        self.z_layer = tf.keras.layers.Lambda(sample, name="z")
        if recurrent:
            if len(n_hidden) > 1:
                if cell_type == "LSTM":
                    encoder = tf.keras.experimental.PeepholeLSTMCell(n_hidden[0], activation=tf.nn.tanh, name="encoder")
                    decoder = tf.keras.experimental.PeepholeLSTMCell(n_hidden[0], activation=tf.nn.tanh, name="decoder")
                else:
                    encoder = tf.keras.layers.GRUCell(n_hidden[0], activation=tf.nn.tanh, name="encoder")
                    decoder = tf.keras.layers.GRUCell(n_hidden[0], activation=tf.nn.tanh, name="decoder")
                self.ae_encode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in n_hidden[1:-1]]
                self.mu_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="mu_tilde")
                self.log_sigma2_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="log_sigma2_tilde")
                self.rnn_transform_layer = RnnDecodeTransformLayer(n_hidden[0], I)
                self.ae_decode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in (n_hidden[:-1])[::-1]]
            else:
                n_hidden = n_hidden[0]
                if cell_type == "LSTM":
                    # one n_hidden for mu, one for sigma2
                    encoder = tf.keras.experimental.PeepholeLSTMCell(
                        n_hidden + n_hidden, activation=tf.nn.tanh, name="encoder")
                    decoder = tf.keras.experimental.PeepholeLSTMCell(n_hidden, activation=tf.nn.tanh, name="decoder")
                else:
                    # one n_hidden for mu, one for sigma2
                    encoder = tf.keras.layers.GRUCell(n_hidden + n_hidden, activation=tf.nn.tanh, name="encoder")
                    decoder = tf.keras.layers.GRUCell(n_hidden, activation=tf.nn.tanh, name="decoder")
                self.mu_layer = tf.keras.layers.Lambda(lambda hidden: hidden[:, :n_hidden], name="mu_tilde")
                self.log_sigma2_layer = tf.keras.layers.Lambda(
                    lambda hidden: hidden[:, n_hidden:], name="log_sigma2_tilde")
                self.rnn_transform_layer = RnnDecodeTransformLayer(n_hidden, I)
            # select the carry state (not the memory state)
            # unroll = True raises "ValueError: Cannot unroll a RNN if the time dimension is undefined."
            self.encoder_rnn = tf.keras.layers.RNN(encoder, unroll=False, return_state=True, return_sequences=True)
            # select the output
            self.decoder_rnn = tf.keras.layers.RNN(decoder, unroll=False, return_state=True, return_sequences=True)
        else:
            self.ae_encode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in n_hidden[:-1]]
            self.mu_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="mu_tilde")
            self.log_sigma2_layer = tf.keras.layers.Dense(n_hidden[-1], activation=None, name="log_sigma2_tilde")
            self.ae_decode_layers = [tf.keras.layers.Dense(n, activation=tf.nn.softplus) for n in (n_hidden[:-1])[::-1]]
            self.x_raw_layer = tf.keras.layers.Dense(D, activation=None, name="x_raw")
    @tf.function
    def encode(self, X, W):
        X_imputed = self.imputation_layer(X, W)
        if self.recurrent:
            # X_imputed = [tf.squeeze(t, [1]) for t in tf.split(X_imputed, self.D, 1)]
            hidden = self.encoder_rnn(X_imputed)[-1] # [1] ???
        else:
            hidden = X_imputed
        if len(self.n_hidden) > 1:
            for layer in self.ae_encode_layers:
                hidden = layer(hidden)
        mu_tilde = self.mu_layer(hidden)
        log_sigma2_tilde = self.log_sigma2_layer(hidden)
        z = self.z_layer((mu_tilde, log_sigma2_tilde))
        return z, mu_tilde, log_sigma2_tilde

    @tf.function
    def decode(self, z):
        hidden = z
        if len(self.n_hidden) > 1:
            for layer in self.ae_decode_layers:
                hidden = layer(hidden)
        if self.recurrent:
            inputs = tf.zeros((tf.shape(z)[0], self.D, self.I))
            initial_state = (tf.zeros(tf.shape(input=hidden)), hidden)
            rnn_output = self.decoder_rnn(inputs=inputs, initial_state=initial_state)[0]
            x_raw = self.rnn_transform_layer(rnn_output, tf.shape(z)[0])
        else:
            x_raw = self.x_raw_layer(hidden)
        x = self.output_activation(x_raw, name="x_output")
        return x, x_raw

    @tf.function
    def call(self, inputs, training=False):
        X = inputs[0]
        W = inputs[1]
        z, mu_tilde, log_sigma2_tilde = self.encode(X, W)
        x, x_raw = self.decode(z)
        dummy_val = tf.constant(0.0, dtype=tf.float32)
        mu_c, sigma2_c, phi_c = self.gmm_layer(dummy_val)
        return x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde
