import tensorflow as tf
from functools import partial
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.stats import norm
import sys
import os
import numpy as np
from sklearn.mixture import GaussianMixture
import multiprocessing as mp
from tensorflow.python import debug as tf_debug

class VADER:
    '''
        A VADER object represents a (recurrent) (variational) (Gaussian mixture) autoencoder
    '''
    def __init__(self, X_train, W_train=None, y_train=None, n_hidden=[12, 2], k=3, output_activation=None,
        batch_size = 32, learning_rate=1e-3, alpha=1.0, phi=None, cell_type="LSTM", recurrent=True,
        save_path=os.path.join('vader', 'vader.ckpt'), eps=1e-10, seed=None, n_thread=0):
        '''
            Constructor for class VADER

            Parameters
            ----------
            X_train : float
                The data to be clustered. Numpy array with dimensions [samples, time points, variables] if recurrent is
                True, else [samples, variables].
            W_train : integer
                Missingness indicator. Numpy array with same dimensions as X_train. Entries in X_train for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored. If None, then no missingness is assumed. (default: None)
            y_train : int
                Cluster labels. Numpy array or list of length X_train.shape[0]. y_train is used purely for monitoring
                performance when a ground truth clustering is available. It does not affect training, and can be omitted
                 if no ground truth is available. (default: None)
            n_hidden : int
                The hidden layers. List of length >= 1. Specification of the number of nodes in the hidden layers.
                (default: [12, 2])
            k : int
                Number of mixture components. (default: 3)
            output_activation : str
                Output activation function, "sigmoid" for binary output, None for continuous output. (default: None)
            batch_size : int
                Batch size used for training. (default: 32)
            learning_rate : float
                Learning rate for training. (default: 1e-3)
            alpha : float
                Weight of the latent loss, relative to the reconstruction loss. (default: 1.0, i.e. equal weight)
            phi : float
                Initial values for the mixture component probabilities. List of length k. If None, then initialization
                is according to a uniform distribution. (default: None)
            cell_type : str
                Cell type of the recurrent neural network. Currently only LSTM is supported. (default: "LSTM")
            recurrent : bool
                Train a recurrent autoencoder, or a non-recurrent autoencoder? (default: True)
            save_path : str
                Location to store the Tensorflow checkpoint files. (default: os.path.join('vader', 'vader.ckpt'))
            eps : float
                Small value used for numerical stability in logarithmic computations, divisions, etc. (default: 1e-10)
            seed : int
                Random seed, to be used for reproducibility. (default: None)
            n_thread : int
                Number of threads, passed to Tensorflow's intra_op_parallelism_threads and inter_op_parallelism_threads.
                (default: 0)

            Attributes
            ----------
            X : float
                The data to be clustered. Numpy array.
            W : int
                Missingness indicator. Numpy array of same dimensions as X_train. Entries in X_train for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored.
            y : int
                Cluster labels. Numpy array or list of length X_train.shape[0]. y_train is used purely for monitoring
                performance when a ground truth clustering is available. It does not affect training, and can be omitted
                 if no ground truth is available.
            n_hidden : int
                The hidden layers. List of length >= 1. Specification of the number of nodes in the hidden layers.
            K : int
                Number of mixture components.
            output_activation : str
                Output activation function, "sigmoid" for binary output, None for continuous output.
            batch_size : int
                Batch size used for training.
            learning_rate : float
                Learning rate for training.
            alpha : float
                Weight of the latent loss, relative to the reconstruction loss.
            phi : float
                Initial values for the mixture component probabilities. List of length k.
            cell_type : str
                Cell type of the recurrent neural network. Currently only LSTM is supported.
            recurrent : bool
                Train a recurrent autoencoder, or a non-recurrent autoencoder?
            save_path : str
                Location to store the Tensorflow checkpoint files.
            eps : float
                Small value used for numerical stability in logarithmic computations, divisions, etc.
            seed : int
                Random seed, to be used for reproducibility.
            n_thread : int
                Number of threads, passed to Tensorflow's intra_op_parallelism_threads and inter_op_parallelism_threads.
            n_epoch : int
                The number of epochs that this VADER object was trained.
            loss : float
                The current training loss of this VADER object.
            n_param : int
                The total number of parameters of this VADER object.
            latent_loss : float
                The current training latent loss of this VADER object.
            reconstruction_loss : float
                The current training reconstruction loss of this VADER object.
            D : int
                self.X.shape[1]. The number of time points if self.recurrent is True, otherwise the number of variables.
            I : integer
                X_train.shape[2]. The number of variables if self.recurrent is True, otherwise not defined.
        '''
        if seed is not None:
            np.random.seed(seed)

        self.D = X_train.shape[1]  # dimensionality of input/output
        self.X = X_train
        if W_train is not None:
            self.W = W_train
        else:
            self.W = np.ones(X_train.shape, dtype=np.float32)
        if y_train is not None:
            self.y = np.asarray(y_train, np.int32)
        else:
            self.y = y_train
        self.save_path = save_path
        self.eps = eps
        self.alpha = alpha  # weight for the latent loss (alpha times the reconstruction loss weight)
        self.learning_rate = learning_rate
        self.K = k  # 10 number of mixture components (clusters)
        self.n_hidden = n_hidden  # n_hidden[-1] is dimensions of the mixture distribution (size of hidden layer)
        if output_activation is None:
            self.output_activation = tf.identity
        else:
            if output_activation == "sigmoid":
                self.output_activation = tf.nn.sigmoid
        self.n_hidden = n_hidden
        self.seed = seed
        self.n_epoch = 0
        self.n_thread = n_thread
        self.batch_size = batch_size
        self.loss = np.array([])
        self.reconstruction_loss = np.array([])
        self.latent_loss = np.array([])
        self.n_param = None
        self.cell_type = cell_type

        self.recurrent = recurrent
        if self.recurrent:
            self.I = X_train.shape[2]  # multivariate dimensions

        if self.recurrent:
            # encode
            def g_monolayer(X):
                n_hidden = self.n_hidden[0]
                X = [tf.squeeze(t, [1]) for t in tf.split(X, self.D, 1)]
                if self.cell_type == "LSTM":
                    encoder = tf.nn.rnn_cell.LSTMCell(
                        num_units=n_hidden + n_hidden,  # one for mu, one for sigma
                        activation=tf.nn.tanh, name="encoder", use_peepholes=True
                    )
                    (_, (_, hidden)) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
                else:
                    encoder = tf.nn.rnn_cell.GRUCell(
                        num_units=n_hidden + n_hidden,  # one for mu, one for sigma
                        activation=tf.nn.tanh, name="encoder"
                    )
                    (_, hidden) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)

                mu = tf.identity(hidden[:, :n_hidden], name="mu_tilde")
                log_sigma = tf.identity(hidden[:, n_hidden:], name="log_sigma_tilde")
                return mu, log_sigma

            # encode
            def g_multilayer(X):
                X = [tf.squeeze(t, [1]) for t in tf.split(X, self.D, 1)]
                if self.cell_type == "LSTM":
                    encoder = tf.nn.rnn_cell.LSTMCell(
                        num_units=self.n_hidden[0],
                        activation=tf.nn.tanh, name="encoder", use_peepholes=True
                    )
                    (_, (_, hidden)) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
                else:
                    encoder = tf.nn.rnn_cell.GRUCell(
                        num_units=self.n_hidden[0],
                        activation=tf.nn.tanh, name="encoder"
                    )
                    (_, hidden) = tf.nn.static_rnn(encoder, X, dtype=tf.float32)
                for n in self.n_hidden[1:-1]:
                    hidden = my_dense_layer(hidden, n)
                mu = tf.identity(my_dense_layer(hidden, self.n_hidden[-1], activation=None), name="mu_tilde")
                log_sigma = tf.identity(my_dense_layer(hidden, self.n_hidden[-1], activation=None),
                                        name="log_sigma_tilde")
                return mu, log_sigma

            # decode
            def f_monolayer(z):

                # inputs = tf.tile(tf.expand_dims(tf.zeros(tf.shape(z)), 1), [int(1), int(self.D), int(1)])
                #
                # # Use LSTM with peepholes instead? --> remember that LSTM initial_state requires a tuple!!!
                # decoder = tf.nn.rnn_cell.GRUCell(
                #     num_units=self.n_hidden[-1],
                #     activation=None, name="decoder"
                # )
                # dec_output, _ = tf.nn.dynamic_rnn(
                #     cell=decoder, inputs=inputs,
                #     dtype=tf.float32, swap_memory=True,
                #     initial_state=z
                # )
                # # # reverse (for easier fitting)
                # # dec_output = tf.reverse(dec_output, axis=[1])
                #
                # # map back to dimensions (D, I)
                # dec_weight = tf.Variable(tf.truncated_normal([self.n_hidden[-1], self.I], dtype=tf.float32))
                # dec_bias = tf.Variable(tf.constant(0.1, shape=[self.I], dtype=tf.float32))
                # x_raw = tf.map_fn(lambda d_o: tf.matmul(d_o, dec_weight), dec_output) + dec_bias
                # x_raw = tf.identity(x_raw, name="x_raw")
                # x = self.output_activation(x_raw, name="x_output")

                n_hidden = self.n_hidden[0]
                dec_weight_ = tf.Variable(tf.truncated_normal([n_hidden, self.I], dtype=tf.float32), name='dec_weight')
                dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.I], dtype=tf.float32), name='dec_bias')

                dec_inputs = [tf.zeros((tf.shape(z)[0], self.I), dtype=tf.float32) for _ in range(self.D)]
                if self.cell_type == "LSTM":
                    _dec_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, name="dec_cell", use_peepholes=True)
                    (dec_outputs, (_, dec_state)) = tf.nn.static_rnn(_dec_cell, dec_inputs, initial_state=(tf.zeros(tf.shape(z)), z), dtype=tf.float32)
                else:
                    _dec_cell = tf.nn.rnn_cell.GRUCell(n_hidden, name="dec_cell")
                    (dec_outputs, dec_state) = tf.nn.static_rnn(_dec_cell, dec_inputs, initial_state=z, dtype=tf.float32)
                # if reverse:
                #     dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [tf.shape(z)[0], 1, 1])
                output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
                x_raw = tf.identity(output_, name="x_raw")
                x = self.output_activation(x_raw, name="x_output")

                return x, x_raw

            # decode
            def f_multilayer(z):
                n_hidden = self.n_hidden[::-1]
                hidden = z
                for n in n_hidden[1:]:
                    hidden = my_dense_layer(hidden, n)
                dec_weight_ = tf.Variable(tf.truncated_normal([n_hidden[-1], self.I], dtype=tf.float32),
                                          name='dec_weight')
                dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.I], dtype=tf.float32), name='dec_bias')

                dec_inputs = [tf.zeros((tf.shape(hidden)[0], self.I), dtype=tf.float32) for _ in range(self.D)]
                if self.cell_type == "LSTM":
                    _dec_cell = tf.nn.rnn_cell.LSTMCell(n_hidden[-1], name="dec_cell", use_peepholes=True)
                    (dec_outputs, (_, dec_state)) = tf.nn.static_rnn(_dec_cell, dec_inputs, initial_state=(tf.zeros(tf.shape(hidden)), hidden), dtype=tf.float32)
                else:
                    _dec_cell = tf.nn.rnn_cell.GRUCell(n_hidden[-1], name="dec_cell")
                    (dec_outputs, dec_state) = tf.nn.static_rnn(_dec_cell, dec_inputs, initial_state=hidden, dtype=tf.float32)

                # if reverse:
                #     dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [tf.shape(hidden)[0], 1, 1])
                output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_
                x_raw = tf.identity(output_, name="x_raw")
                x = self.output_activation(x_raw, name="x_output")
                return x, x_raw

            def g(X):
                if len(self.n_hidden) > 1:
                    return g_multilayer(X)
                else:
                    return g_monolayer(X)

            def f(X):
                if len(self.n_hidden) > 1:
                    return f_multilayer(X)
                else:
                    return f_monolayer(X)
        else:
            # encode
            def g(X):
                hidden = tf.clip_by_value(X, self.eps, 1 - self.eps)
                for n in self.n_hidden[:-1]:
                    hidden = my_dense_layer(hidden, n)
                mu = tf.identity(my_dense_layer(hidden, self.n_hidden[-1], activation=None), name="mu_tilde")
                log_sigma = tf.identity(my_dense_layer(hidden, self.n_hidden[-1], activation=None), name="log_sigma_tilde")
                return mu, log_sigma

            # decode
            def f(z):
                hidden = z
                for n in (self.n_hidden[:-1])[::-1]:
                    hidden = my_dense_layer(hidden, n)
                x_raw = tf.identity(my_dense_layer(hidden, self.D, activation=None), name="x_raw")
                x = self.output_activation(x_raw, name="x_output")  # the reconstructions, one for each mixture component
                return x, x_raw

        # initializer = tf.contrib.layers.variance_scaling_initializer()
        # l2_reg = 0.0001
        # he_init = tf.contrib.layers.variance_scaling_initializer()  # He initialization
        # l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        my_dense_layer = partial(
            tf.layers.dense,
            activation=tf.nn.softplus,  # tf.nn.softplus, # tf.nn.elu,
            kernel_initializer=None  # initializer
        )

        def reconstruction_loss(X, x, x_raw, W):
            # reconstruction loss: E[log p(x|z)]

            if (self.output_activation == tf.nn.sigmoid):
                rec_loss = tf.losses.sigmoid_cross_entropy(tf.clip_by_value(X, self.eps, 1 - self.eps), x_raw, W)
            else:
                rec_loss = tf.losses.mean_squared_error(X, x, W)

            # re-scale the loss to the original dims (making sure it balances correctly with the latent loss)
            rec_loss = rec_loss * tf.cast(tf.reduce_prod(tf.shape(W)), dtype=tf.float32) / tf.reduce_sum(W)
            if self.recurrent:
                # sum across the features, average across the samples
                rec_loss = self.D * self.I * rec_loss
            else:
                # sum across the features, average across the samples
                rec_loss = self.D * rec_loss
            # rec_loss = self.D * rec_loss

            return rec_loss

        def latent_loss(z, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde):
            if self.K == 1: # ordinary VAE
                latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(
                    tf.square(tf.exp(log_sigma_tilde)) + tf.square(mu_tilde) - 1 - 2 * log_sigma_tilde,
                    axis=1
                ))
            else:
                # latent_loss = 0.5 * tf.reduce_sum(
                #     tf.square(tf.exp(log_sigma_tilde)) + tf.square(mu_tilde)
                #     - 1 - tf.log(eps + tf.square(tf.exp(log_sigma_tilde))))
                # latent_loss = tf.identity(latent_loss, name = "latent_loss")

                def log_pdf(z, mu, sigma):
                    def f(i):
                        return - tf.square(z - mu[i]) / 2.0 / (self.eps + sigma[i]) - tf.log(
                            self.eps + self.eps + 2.0 * np.pi * sigma[i]) / 2.0
                    return tf.transpose(tf.map_fn(f, np.arange(self.K), dtype=tf.float32), [1, 0, 2])

                # log_p = tf.reduce_sum(tf.log(phi_c) - 0.5 * tf.log(2 * np.pi * sigma_c))

                # log_p = tf.log(self.eps + phi_c) + tf.reduce_sum(log_pdf(z, mu_c, sigma_c), axis=2)
                # gamma_c = tf.nn.softmax(tf.exp(log_p))
                # log_gamma_c = tf.log(gamma_c + self.eps)

                log_p = tf.log(self.eps + phi_c) + tf.reduce_sum(log_pdf(z, mu_c, sigma_c), axis=2)
                lse_p = tf.reduce_logsumexp(log_p, keepdims=True, axis=1)
                log_gamma_c = log_p - lse_p

                gamma_c = tf.exp(log_gamma_c)

                # latent loss: E[log p(z|c) + log p(c) - log q(z|x) - log q(c|x)]
                term1 = tf.log(self.eps + sigma_c)
                term2 = tf.transpose(
                    tf.map_fn(lambda i: tf.exp(log_sigma_tilde) / (self.eps + sigma_c[i]), np.arange(self.K), tf.float32),
                    [1, 0, 2])
                term3 = tf.transpose(
                    tf.map_fn(lambda i: tf.square(mu_tilde - mu_c[i]) / (self.eps + sigma_c[i]), np.arange(self.K),
                              tf.float32), [1, 0, 2])

                latent_loss1 = 0.5 * tf.reduce_sum(gamma_c * tf.reduce_sum(term1 + term2 + term3, axis=2), axis=1)
                # latent_loss2 = - tf.reduce_sum(gamma_c * tf.log(self.eps + phi_c / (self.eps + gamma_c)), axis=1)
                latent_loss2 = - tf.reduce_sum(gamma_c * (tf.log(self.eps + phi_c) - log_gamma_c), axis=1)
                latent_loss3 = - 0.5 * tf.reduce_sum(1 + log_sigma_tilde, axis=1)
                # average across the samples
                latent_loss1 = tf.reduce_mean(latent_loss1)
                latent_loss2 = tf.reduce_mean(latent_loss2)
                latent_loss3 = tf.reduce_mean(latent_loss3)
                # add the different terms
                latent_loss = latent_loss1 + latent_loss2 + latent_loss3
            return latent_loss

        tf.reset_default_graph()
        graph = tf.get_default_graph()

        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            if self.recurrent:
                X = tf.placeholder(tf.float32, [None, self.D, self.I], name="X_input")
                W = tf.placeholder(tf.float32, [None, self.D, self.I], name="W_input")
                A = tf.get_variable("A", shape=[self.D, self.I], dtype=tf.float32, trainable=True)
            else:
                X = tf.placeholder(tf.float32, [None, self.D], name="X_input")
                W = tf.placeholder(tf.float32, [None, self.D], name="W_input")
                A = tf.get_variable("A", shape=[self.D], dtype=tf.float32, trainable=True)
            alpha_t = tf.placeholder_with_default(tf.convert_to_tensor(self.alpha), (), name="alpha_input")
            mu_c_unscaled = tf.get_variable("mu_c_unscaled", [self.K, self.n_hidden[-1]], dtype=tf.float32, trainable=True)
            mu_c = tf.identity(mu_c_unscaled, name="mu_c")
            sigma_c_unscaled = tf.get_variable("sigma_c_unscaled", shape=[self.K, self.n_hidden[-1]], dtype=tf.float32, trainable=True)
            sigma_c = tf.nn.softplus(sigma_c_unscaled, name="sigma_c")
            if phi is None:
                phi_c_unscaled = tf.get_variable("phi_c_unscaled", shape=[self.K], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(1))
            else: # set phi_c to some constant provided by the user
                phi_c_unscaled = tf.get_variable("phi_c_unscaled", dtype=tf.float32, trainable=False, initializer=tf.log(tf.constant(phi)))
            phi_c = tf.nn.softmax(phi_c_unscaled, name="phi_c")

            # encode
            # Treat W as an indicator for nonmissingness (1: nonmissing; 0: missing)
            if ~np.all(self.W == 1.0) and np.all(np.logical_or(self.W == 0.0, self.W == 1.0)):
                XW = tf.multiply(X, W) + tf.multiply(A, (1.0 - W))
                mu_tilde, log_sigma_tilde = g(XW)
            else:
                mu_tilde, log_sigma_tilde = g(X)

            # sample from the mixture component
            noise = tf.random_normal(tf.shape(log_sigma_tilde), dtype=tf.float32)
            z = tf.add(mu_tilde, tf.exp(log_sigma_tilde / 2) * noise, name="z")

            # decode
            x, x_raw = f(z)

            # calculate the loss
            rec_loss = reconstruction_loss(X, x, x_raw, W)
            rec_loss = tf.identity(rec_loss, name="reconstruction_loss")

            lat_loss = tf.cond(
                tf.greater(alpha_t, tf.convert_to_tensor(0.0)),
                lambda: tf.multiply(alpha_t, latent_loss(z, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde)), # variational
                lambda: tf.convert_to_tensor(0.0) # non-variational
            )
            lat_loss = tf.identity(lat_loss, name="latent_loss")

            loss = tf.add(rec_loss, lat_loss, name="loss")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, name="optimizer")
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            training_op = optimizer.apply_gradients(zip(gradients, variables), name="training_op")

            init = tf.global_variables_initializer()
            self.n_param = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
            saver = tf.train.Saver()

        with tf.Session(graph=graph, config=tf.ConfigProto(
            intra_op_parallelism_threads=self.n_thread, inter_op_parallelism_threads=self.n_thread)) as sess:
            init.run()
            saver.save(sess, self.save_path)

    def _restore_session(self):
        tf.reset_default_graph()
        sess = tf.InteractiveSession(config=tf.ConfigProto(
            intra_op_parallelism_threads=self.n_thread,
            inter_op_parallelism_threads=self.n_thread
        ))
        saver = tf.train.import_meta_graph(self.save_path + ".meta")
        saver.restore(sess, self.save_path)
        graph = tf.get_default_graph()
        return sess, saver, graph

    def _cluster(self, mu_t, mu, sigma, phi):
        def f(mu_t, mu, sigma, phi):
            # the covariance matrix is diagonal, so we can just take the product
            p = np.log(self.eps + phi) + np.sum(np.log(self.eps + norm.pdf(mu_t, loc=mu, scale=np.sqrt(sigma))), axis=1)
            return np.argmax(p)
        return np.array([f(mu_t[i], mu, sigma, phi) for i in np.arange(mu_t.shape[0])])

    def _accuracy(self, y_pred, y_true):
        def cluster_acc(Y_pred, Y):
            assert Y_pred.size == Y.size
            D = max(Y_pred.max(), Y.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(Y_pred.size):
                w[Y_pred[i], Y[i]] += 1
            ind = linear_assignment(w.max() - w)
            return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, np.array(w)

        return cluster_acc(y_pred, y_true)

    def _get_vars(self, graph):
        training_op = graph.get_operation_by_name("training_op")
        loss = graph.get_tensor_by_name("loss:0")
        rec_loss = graph.get_tensor_by_name("reconstruction_loss:0")
        lat_loss = graph.get_tensor_by_name("latent_loss:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        alpha_t = graph.get_tensor_by_name("alpha_input:0")
        z = graph.get_tensor_by_name("z:0")
        x_output = graph.get_tensor_by_name("x_output:0")
        sigma_c = graph.get_tensor_by_name("sigma_c:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")
        mu_tilde = graph.get_tensor_by_name("mu_tilde:0")
        log_sigma_tilde = graph.get_tensor_by_name("log_sigma_tilde:0")
        phi_c = graph.get_tensor_by_name("phi_c:0")
        return training_op, loss, rec_loss, lat_loss, X, W, alpha_t, z, x_output, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde

    def _get_batch(self, batch_size):
        ii = np.random.choice(np.arange(self.X.shape[0]), batch_size, replace=False)
        X_batch = self.X[ii,]
        if self.y is not None:
            y_batch = self.y[ii]
        else:
            y_batch = None
        W_batch = self.W[ii,]
        return X_batch, y_batch, W_batch

    def _print_progress(self, epoch, sess, graph):
        X_batch, y_batch, W_batch = self._get_batch(max(10 * self.batch_size, self.X.shape[0]))
        training_op, loss, rec_loss, lat_loss, X, W, alpha_t, z, x_output, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde = self._get_vars(
            graph)
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, rec_loss, lat_loss],
                                                                      feed_dict={X: X_batch, W: W_batch, alpha_t: self.alpha})
        self.reconstruction_loss = np.append(self.reconstruction_loss, reconstruction_loss_val)
        self.latent_loss = np.append(self.latent_loss, latent_loss_val)
        self.loss = np.append(self.loss, loss_val)
        clusters = self._cluster(mu_tilde.eval(feed_dict={X: X_batch, W: W_batch, alpha_t: self.alpha}), mu_c.eval(), sigma_c.eval(),
                                 phi_c.eval())
        # if y_batch is not None:
        if y_batch is not None:
            acc, _ = self._accuracy(clusters, y_batch)
            print(epoch,
                  "tot_loss:", "%.2f" % round(loss_val, 2),
                  "\trec_loss:", "%.2f" % round(reconstruction_loss_val, 2),
                  "\tlat_loss:", "%.2f" % round(latent_loss_val, 2),
                  "\tacc:", "%.2f" % round(acc, 2),
                  flush=True
                  )
        else:
            print(epoch,
                  "tot_loss:", "%.2f" % round(loss_val, 2),
                  "\trec_loss:", "%.2f" % round(reconstruction_loss_val, 2),
                  "\tlat_loss:", "%.2f" % round(latent_loss_val, 2),
                  flush=True
                  )
        return 0

    def fit(self, n_epoch=10, verbose=False):
        '''
            Train a VADER object.

            Parameters
            ----------
            n_epoch : int
                Train n_epoch epochs. (default: 10)
            verbose : bool
                Print progress? (default: False)

            Returns
            -------
            0 if successful
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        self.n_epoch += n_epoch

        sess, saver, graph = self._restore_session()
        writer = tf.summary.FileWriter(self.save_path, sess.graph)
        training_op, loss, rec_loss, lat_loss, X, W, alpha_t, z, x_output, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde = self._get_vars(graph)
        X = graph.get_tensor_by_name("X_input:0")
        if verbose:
            self._print_progress(-1, sess, graph)
        for epoch in range(n_epoch): # NOTE: explicitly not self.epoch in case of repeated calls to fit!
            n_batches = self.X.shape[0] // self.batch_size
            for iteration in range(n_batches):
                sys.stdout.flush()
                X_batch, y_batch, W_batch = self._get_batch(self.batch_size)
                sess.run(training_op, feed_dict={X: X_batch, W: W_batch, alpha_t: self.alpha})
            if verbose:
                self._print_progress(epoch, sess, graph)
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, rec_loss, lat_loss], feed_dict={X: self.X, W: self.W, alpha_t: self.alpha})
        self.reconstruction_loss = np.append(self.reconstruction_loss, reconstruction_loss_val)
        self.latent_loss = np.append(self.latent_loss, latent_loss_val)
        self.loss = np.append(self.loss, loss_val)
        saver.save(sess, self.save_path)
        writer.close()
        sess.close()
        return 0

    def pre_fit(self, n_epoch=10, verbose=False):
        '''
            Pre-train a VADER object using only the latent loss, and initialize the Gaussian mixture parameters using
            the resulting latent representation.

            Parameters
            ----------
            n_epoch : int
                Train n_epoch epochs. (default: 10)
            verbose : bool
                Print progress? (default: False)

            Returns
            -------
            0 if successful
        '''
        # save the alpha
        alpha = self.alpha
        # pre-train using non-variational AEs
        self.alpha = 0.0

        # pre-train
        ret = self.fit(n_epoch, verbose)

        # map to latent
        z = self.map_to_latent(self.X, self.W, n_samp=10)
        # fit GMM
        gmm = GaussianMixture(n_components=self.K, covariance_type="diag").fit(z)
        # get GMM parameters
        phi = np.log(gmm.weights_ + self.eps) # inverse softmax
        mu = gmm.means_
        sigma = np.log(np.exp(gmm.covariances_) - 1.0 + self.eps) # inverse softplus

        # initialize mixture components
        sess, saver, graph = self._restore_session()
        def my_get_variable(varname):
            return [v for v in tf.global_variables() if v.name == varname][0]
        mu_c_unscaled = my_get_variable("mu_c_unscaled:0")
        sess.run(mu_c_unscaled.assign(tf.convert_to_tensor(mu, dtype=tf.float32)))
        sigma_c_unscaled = my_get_variable("sigma_c_unscaled:0")
        sess.run(sigma_c_unscaled.assign(tf.convert_to_tensor(sigma, dtype=tf.float32)))
        phi_c_unscaled = my_get_variable("phi_c_unscaled:0")
        sess.run(phi_c_unscaled.assign(tf.convert_to_tensor(phi, dtype=tf.float32)))
        saver.save(sess, self.save_path)
        sess.close()

        # restore the alpha
        self.alpha = alpha
        return ret

    def map_to_latent(self, X_c, W_c=None, n_samp=1):
        '''
            Map an input to its latent representation.

            Parameters
            ----------
            X_c : float
                The data to be clustered. Numpy array with dimensions [samples, time points, variables] if recurrent is
                True, else [samples, variables]
            W_c : int
                Missingness indicator. Numpy array with same dimensions as X_c. Entries in X_c for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored. If None, then no missingness is assumed. (default: None)
            n_samp : int
                The number of latent samples to take for each input sample. (default: 1)

            Returns
            -------
            numpy array containing the latent representations.
        '''
        if W_c is None:
            W_c = np.ones(X_c.shape)
        sess, saver, graph = self._restore_session()
        z = graph.get_tensor_by_name("z:0")
        mu_tilde = graph.get_tensor_by_name("mu_tilde:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        latent = np.concatenate([z.eval(feed_dict={X: X_c, W: W_c}) for i in np.arange(n_samp)], axis=0)
        sess.close()
        return latent

    def get_loss(self, X_c, W_c=None, mu_c=None, sigma_c=None, phi_c=None):
        '''
            Calculate the loss for specific input data and Gaussian mixture parameters.

            Parameters
            ----------
            X_c : float
                The data to be clustered. Numpy array with dimensions [samples, time points, variables] if recurrent is
                True, else [samples, variables]
            W_c : int
                Missingness indicator. Numpy array with same dimensions as X_c. Entries in X_c for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored. If None, then no missingness is assumed. (default: None)
            mu_c : float
                The mixture component means, one for each self.K. Numpy array of dimensions [self.K, self.n_hidden[-1]].
                If None, then the means of this VADER object are used. (default: None)
            sigma_c : float
                The mixture component variances, representing diagonal covariance matrices, one for each self.K. Numpy
                array of dimensions [self.K, self.n_hidden[-1]]. If None, then the variances of this VADER object are
                used. (default: None)
            phi_c : float
                The mixture component probabilities. List of length self.K. If None, then the component probabilities of
                 this VADER object are used. (default: None)

            Returns
            -------
            Dictionary with two components, "reconstruction_loss" and "latent_loss".
        '''
        if W_c is None:
            W_c = np.ones(X_c.shape, dtype=np.float32)
        sess, saver, graph = self._restore_session()
        rec_loss = graph.get_tensor_by_name("reconstruction_loss:0")
        lat_loss = graph.get_tensor_by_name("latent_loss:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        alpha_t = graph.get_tensor_by_name("alpha_input:0")

        def my_get_variable(varname):
            return [v for v in tf.global_variables() if v.name == varname][0]

        mu_c_unscaled = my_get_variable("mu_c_unscaled:0")
        if mu_c is not None:
            mu_c_new = mu_c
            mu_c_old = mu_c_unscaled.eval()
            sess.run(mu_c_unscaled.assign(tf.convert_to_tensor(mu_c_new, dtype=tf.float32)))

        sigma_c_unscaled = my_get_variable("sigma_c_unscaled:0")
        if sigma_c is not None:
            sigma_c_new = sigma_c
            sigma_c_old = sigma_c_unscaled.eval()
            sess.run(sigma_c_unscaled.assign(tf.contrib.distributions.softplus_inverse(tf.convert_to_tensor(sigma_c_new, dtype=tf.float32))))

        phi_c_unscaled = my_get_variable("phi_c_unscaled:0")
        if phi_c is not None:
            phi_c_new = phi_c
            phi_c_old = phi_c_unscaled.eval()
            sess.run(phi_c_unscaled.assign(tf.convert_to_tensor(phi_c_new, dtype=tf.float32)))

        lat_loss = lat_loss.eval(feed_dict={X: X_c, W: W_c, alpha_t: self.alpha})
        rec_loss = rec_loss.eval(feed_dict={X: X_c, W: W_c, alpha_t: self.alpha})

        sess.close()
        return {"reconstruction_loss": rec_loss, "latent_loss": lat_loss}

    def cluster(self, X_c, W_c=None, mu_c=None, sigma_c=None, phi_c=None):
        '''
            Cluster input data using this VADER object.

            Parameters
            ----------
            X_c : float
                The data to be clustered. Numpy array with dimensions [samples, time points, variables] if recurrent is
                True, else [samples, variables]
            W_c : int
                Missingness indicator. Numpy array with same dimensions as X_c. Entries in X_c for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored. If None, then no missingness is assumed. (default: None)
            mu_c : float
                The mixture component means, one for each self.K. Numpy array of dimensions [self.K, self.n_hidden[-1]].
                If None, then the means of this VADER object are used. (default: None)
            sigma_c : float
                The mixture component variances, representing diagonal covariance matrices, one for each self.K. Numpy
                array of dimensions [self.K, self.n_hidden[-1]]. If None, then the variances of this VADER object are
                used. (default: None)
            phi_c : float
                The mixture component probabilities. List of length self.K. If None, then the component probabilities of
                 this VADER object are used. (default: None)

            Returns
            -------
            Clusters encoded as integers.
        '''
        sess, saver, graph = self._restore_session()
        mu_tilde = graph.get_tensor_by_name("mu_tilde:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        if phi_c is None:
            mu_c = graph.get_tensor_by_name("mu_c:0").eval()
        if phi_c is None:
            sigma_c = graph.get_tensor_by_name("sigma_c:0").eval()
        if phi_c is None:
            phi_c = graph.get_tensor_by_name("phi_c:0").eval()

        if W_c is None:
            W_c = np.ones(X_c.shape)

        clusters = self._cluster(mu_tilde.eval(feed_dict={X: X_c, W: W_c}), mu_c, sigma_c, phi_c)
        sess.close()
        return clusters

    def get_clusters(self):
        '''
            Get the cluster averages represented by this VADER object. Technically, this maps the latent Gaussian
            mixture means to output values using this VADER object.

            Returns
            -------
            Cluster averages.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()
        z = graph.get_tensor_by_name("z:0")
        x = graph.get_tensor_by_name("x_output:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")

        clusters = x.eval(feed_dict={z: mu_c.eval()})
        sess.close()
        return clusters

    def get_latent_distribution(self):
        '''
            Get the parameters of the Gaussian mixture distribution of this VADER object.

            Returns
            -------
            Dictionary with three components, "mu", "sigma_sq", "phi".
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()
        sigma_c = graph.get_tensor_by_name("sigma_c:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")
        phi_c = graph.get_tensor_by_name("phi_c:0")

        res = {
            "mu": mu_c.eval(),
            "sigma_sq": sigma_c.eval(),
            "phi": phi_c.eval()
        }

        sess.close()
        return res

    def generate(self, n):
        '''
            Generate random samples from this VADER object.

            n : int
                The number of samples to generate.

            Returns
            -------
            A dictionary with two components, "clusters" (cluster indicator) and "samples" (the random samples).
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()

        z = graph.get_tensor_by_name("z:0")
        x = graph.get_tensor_by_name("x_output:0")
        sigma_c = graph.get_tensor_by_name("sigma_c:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")
        phi_c = graph.get_tensor_by_name("phi_c:0")

        if self.recurrent:
            gen = np.zeros((n, self.D, self.I), dtype=np.float)
        else:
            gen = np.zeros((n, self.D), dtype=np.float)
        c = np.random.choice(np.arange(self.K), size=n, p=phi_c.eval())
        for k in np.arange(self.K):
            ii = np.flatnonzero(c == k)
            z_rnd = \
                mu_c.eval()[None, k, :] + \
                np.sqrt(sigma_c.eval())[None, k, :] * \
                np.random.normal(size=[ii.shape[0], self.n_hidden[-1]])
            # gen[ii,:] = x.eval(feed_dict={z: z_rnd})
            if self.recurrent:
                gen[ii,:,:] = x.eval(feed_dict={z: z_rnd})
            else:
                gen[ii,:] = x.eval(feed_dict={z: z_rnd})
        sess.close()
        gen = {
            "clusters": c,
            "samples": gen
        }
        return gen

    def predict(self, X_test, W_test=None):
        '''
            Map input data to output (i.e. reconstructed input).

            Parameters
            ----------
            X_test : float numpy array
                The input data to be mapped.
            W_test : integer numpy array of same dimensions as X_test
                Missingness indicator. Entries in X_test for which the corresponding entry in W_test equals 0 are
                treated as missing. More precisely, their specific numeric value is completely ignored.

            Returns
            -------
            numpy array with reconstructed input (i.e. the autoencoder output).
        '''

        if W_test is None:
            W_test = np.ones(X_test.shape)

        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()

        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        x_output = graph.get_tensor_by_name("x_output:0")

        pred = x_output.eval(feed_dict={X: X_test, W: W_test})
        sess.close()
        return pred
