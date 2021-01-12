import tensorflow as tf
import time
from functools import partial
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import multivariate_normal
import sys
import os
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from losses import vader_reconstruction_loss, vader_latent_loss
from layers import encode, decode

class VADER:
    '''
        A VADER object represents a (recurrent) (variational) (Gaussian mixture) autoencoder
    '''
    def __init__(self, X_train, W_train=None, y_train=None, n_hidden=[12, 2], k=3, groups=None, output_activation=None,
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
                The hidden layers. List of length >= 1. Specification of the number of nodes in the hidden layers. For
                example, specifying [a, b, c] will lead to an architecture with layer sizes a -> b -> c -> b -> a.
                (default: [12, 2])
            k : int
                Number of mixture components. (default: 3)
            groups : int
                Grouping of the input variables as a list of length X.shape[2], with integers {0, 1, 2, ...} denoting
                groups; used for weighting proportional to group size. (default: None)
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
            groups: int
                Grouping of the input variables as a list of length self.X.shape[2], with integers {0, 1, 2, ...}
                denoting groups; used for weighting proportional to group size.
            G : float
                Weights determined by variable groups, as computed from the groups argument.
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
        if groups is not None:
            self.groups = np.array(groups, np.int32)
            self.G = 1 / np.bincount(groups)[groups]
            self.G = self.G / sum(self.G)
            self.G = np.broadcast_to(self.G, self.X.shape)
        else:
            self.groups = np.ones(X_train.shape[-1], np.int32)
            self.G = np.ones(X_train.shape, dtype=np.float32)

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
        else:
            self.I = 1

        # encoder function
        def g(X):
            return encode(X, self.D, self.I, self.cell_type, self.n_hidden, self.recurrent)

        # decoder function
        def f(z):
            return decode(z, self.D, self.I, self.cell_type, self.n_hidden, self.recurrent, self.output_activation)

        # reconstruction loss function
        def reconstruction_loss(X, x, x_raw, W):
            return vader_reconstruction_loss(X, x, x_raw, W, self.output_activation, self.D, self.I, self.eps)

        # latent loss function
        def latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde):
            return vader_latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, self.K, self.eps)

        tf.reset_default_graph()
        graph = tf.get_default_graph()

        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            if self.recurrent:
                X = tf.placeholder(tf.float32, [None, self.D, self.I], name="X_input")
                W = tf.placeholder(tf.float32, [None, self.D, self.I], name="W_input")
                G = tf.placeholder(tf.float32, [None, self.D, self.I], name="G_input")
                A = tf.get_variable("A", dtype=tf.float32, trainable=True, initializer=self._initialize_imputation(self.X, self.W))
            else:
                X = tf.placeholder(tf.float32, [None, self.D], name="X_input")
                W = tf.placeholder(tf.float32, [None, self.D], name="W_input")
                G = tf.placeholder(tf.float32, [None, self.D], name="G_input")
                A = tf.get_variable("A", dtype=tf.float32, trainable=True, initializer=self._initialize_imputation(self.X, self.W))
            alpha_t = tf.placeholder_with_default(tf.convert_to_tensor(self.alpha), (), name="alpha_input")
            mu_c_unscaled = tf.get_variable("mu_c_unscaled", [self.K, self.n_hidden[-1]], dtype=tf.float32, trainable=True)
            mu_c = tf.identity(mu_c_unscaled, name="mu_c")
            sigma2_c_unscaled = tf.get_variable("sigma2_c_unscaled", shape=[self.K, self.n_hidden[-1]], dtype=tf.float32, trainable=True)
            sigma2_c = tf.nn.softplus(sigma2_c_unscaled, name="sigma2_c")
            if phi is None:
                phi_c_unscaled = tf.get_variable("phi_c_unscaled", shape=[self.K], dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(1))
            else: # set phi_c to some constant provided by the user
                phi_c_unscaled = tf.get_variable("phi_c_unscaled", dtype=tf.float32, trainable=False, initializer=tf.log(tf.constant(phi)))
            phi_c = tf.nn.softmax(phi_c_unscaled, name="phi_c")

            # encode
            # Treat W as an indicator for nonmissingness (1: nonmissing; 0: missing)
            if ~np.all(self.W == 1.0) and np.all(np.logical_or(self.W == 0.0, self.W == 1.0)):
                XW = tf.multiply(X, W) + tf.multiply(A, (1.0 - W))
                mu_tilde, log_sigma2_tilde = g(XW)
            else:
                mu_tilde, log_sigma2_tilde = g(X)

            # sample from the mixture component
            noise = tf.random_normal(tf.shape(log_sigma2_tilde), dtype=tf.float32)
            z = tf.add(mu_tilde, tf.exp(0.5 * log_sigma2_tilde) * noise, name="z")

            # decode
            x, x_raw = f(z)

            # calculate the loss
            rec_loss = reconstruction_loss(X, x, x_raw, G * W)
            rec_loss = tf.identity(rec_loss, name="reconstruction_loss")

            lat_loss = tf.cond(
                tf.greater(alpha_t, tf.convert_to_tensor(0.0)),
                lambda: tf.multiply(alpha_t, latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde)), # variational
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

    def _initialize_imputation(self, X, W):
        # average per time point, variable
        W_A = np.sum(W, axis=0)
        A = np.sum(X * W, axis=0)
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        # if not available, then average across entire variable
        if self.recurrent:
            for i in np.arange(A.shape[0]):
                for j in np.arange(A.shape[1]):
                    if W_A[i,j] == 0:
                        A[i,j] = np.sum(X[:,:,j]) / np.sum(W[:,:,j])
                        W_A[i,j] = 1
        # if not available, then average across all variables
        A[W_A==0] = np.mean(X[W==1])
        return A.astype(np.float32)
        # return np.zeros(X.shape[1:], np.float32)

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

    # def _cluster(self, mu_t, mu, sigma2, phi):
    #     def f(mu_t, mu, sigma2, phi):
    #         # the covariance matrix is diagonal, so we can just take the product
    #         p = np.log(self.eps + phi) + np.sum(np.log(self.eps + norm.pdf(mu_t, loc=mu, scale=np.sqrt(sigma2))), axis=1)
    #         return np.argmax(p)
    #     return np.array([f(mu_t[i], mu, sigma2, phi) for i in np.arange(mu_t.shape[0])])

    def _cluster(self, mu_t, mu, sigma2, phi):
        # import pickle
        # pickle.dump(mu_t, open( "mu_t.pickle", "wb" ) )
        # pickle.dump(mu, open( "mu.pickle", "wb" ) )
        # pickle.dump(sigma2, open( "sigma2.pickle", "wb" ) )
        # pickle.dump(phi, open( "phi.pickle", "wb" ) )
        def f(mu_t, mu, sigma2, phi):
            # the covariance matrix is diagonal, so we can just take the product
            return np.log(self.eps + phi) + np.log(self.eps + multivariate_normal.pdf(mu_t, mean=mu, cov=np.diag(sigma2)))
        p = np.array([f(mu_t, mu[i], sigma2[i], phi[i]) for i in np.arange(mu.shape[0])])
        return np.argmax(p, axis=0)

    def _accuracy(self, y_pred, y_true):
        def cluster_acc(Y_pred, Y):
            assert Y_pred.size == Y.size
            D = max(Y_pred.max(), Y.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(Y_pred.size):
                w[Y_pred[i], Y[i]] += 1
            ind = np.transpose(np.asarray(linear_assignment(w.max() - w)))
            return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, np.array(w)

        y_pred = np.array(y_pred, np.int32)
        y_true = np.array(y_true, np.int32)
        return cluster_acc(y_pred, y_true)

    def _get_vars(self, graph):
        training_op = graph.get_operation_by_name("training_op")
        loss = graph.get_tensor_by_name("loss:0")
        rec_loss = graph.get_tensor_by_name("reconstruction_loss:0")
        lat_loss = graph.get_tensor_by_name("latent_loss:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        G = graph.get_tensor_by_name("G_input:0")
        alpha_t = graph.get_tensor_by_name("alpha_input:0")
        z = graph.get_tensor_by_name("z:0")
        x_output = graph.get_tensor_by_name("x_output:0")
        sigma2_c = graph.get_tensor_by_name("sigma2_c:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")
        mu_tilde = graph.get_tensor_by_name("mu_tilde:0")
        log_sigma2_tilde = graph.get_tensor_by_name("log_sigma2_tilde:0")
        phi_c = graph.get_tensor_by_name("phi_c:0")
        return training_op, loss, rec_loss, lat_loss, X, W, G, alpha_t, z, x_output, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde

    def _get_batch(self, batch_size):
        ii = np.random.choice(np.arange(self.X.shape[0]), batch_size, replace=False)
        X_batch = self.X[ii,]
        if self.y is not None:
            y_batch = self.y[ii]
        else:
            y_batch = None
        W_batch = self.W[ii,]
        G_batch = self.G[ii,]
        return X_batch, y_batch, W_batch, G_batch

    def _print_progress(self, epoch, sess, graph):
        X_batch, y_batch, W_batch, G_batch = self._get_batch(min(10 * self.batch_size, self.X.shape[0]))
        training_op, loss, rec_loss, lat_loss, X, W, G, alpha_t, z, x_output, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde = self._get_vars(
            graph)
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, rec_loss, lat_loss],
                                                                      feed_dict={X: X_batch, W: W_batch, G: G_batch, alpha_t: self.alpha})
        self.reconstruction_loss = np.append(self.reconstruction_loss, reconstruction_loss_val)
        self.latent_loss = np.append(self.latent_loss, latent_loss_val)
        self.loss = np.append(self.loss, loss_val)
        clusters = self._cluster(mu_tilde.eval(feed_dict={X: X_batch, W: W_batch, G: G_batch, alpha_t: self.alpha}), mu_c.eval(), sigma2_c.eval(),
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

    def fit(self, n_epoch=10, learning_rate=None, verbose=False):
        '''
            Train a VADER object.

            Parameters
            ----------
            n_epoch : int
                Train n_epoch epochs. (default: 10)
            learning_rate: float
                Learning rate for this set of epochs (default: learning rate specified at object construction)
                (NB: not currently used!)
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

        # if learning_rate is not None:
        #     lr_t = graph.get_tensor_by_name("learning_rate:0")

        writer = tf.summary.FileWriter(self.save_path, sess.graph)
        training_op, loss, rec_loss, lat_loss, X, W, G, alpha_t, z, x_output, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde = self._get_vars(graph)
        X = graph.get_tensor_by_name("X_input:0")
        if verbose:
            self._print_progress(-1, sess, graph)
        for epoch in range(n_epoch): # NOTE: explicitly not self.epoch in case of repeated calls to fit!
            n_batches = self.X.shape[0] // self.batch_size
            for iteration in range(n_batches):
                X_batch, y_batch, W_batch, G_batch = self._get_batch(self.batch_size)
                sess.run(training_op, feed_dict={X: X_batch, W: W_batch, G: G_batch, alpha_t: self.alpha})
            if verbose:
                self._print_progress(epoch, sess, graph)
        # loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, rec_loss, lat_loss], feed_dict={X: self.X, W: self.W, G: self.G, alpha_t: self.alpha})
        # self.reconstruction_loss = np.append(self.reconstruction_loss, reconstruction_loss_val)
        # self.latent_loss = np.append(self.latent_loss, latent_loss_val)
        # self.loss = np.append(self.loss, loss_val)
        # if learning_rate is not None:
        #     self.learning_rate = self_learning_rate
        saver.save(sess, self.save_path)
        writer.close()
        sess.close()
        return 0

    def pre_fit(self, n_epoch=10, GMM_initialize=True, learning_rate=None, verbose=False):
        '''
            Pre-train a VADER object using only the latent loss, and initialize the Gaussian mixture parameters using
            the resulting latent representation.

            Parameters
            ----------
            n_epoch : int
                Train n_epoch epochs. (default: 10)
            GMM_initialize: bool
                Should a GMM be fit on the pre-trained latent layer, in order to initialize the VaDER
                mixture component parameters?
            learning_rate: float
                Learning rate for this set of epochs(default: learning rate specified at object construction)
                (NB: not currently used!)
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
        ret = self.fit(n_epoch, learning_rate, verbose)

        if GMM_initialize:
            try:
                # map to latent
                z = self.map_to_latent(self.X, self.W, n_samp=10)
                # fit GMM
                gmm = GaussianMixture(n_components=self.K, covariance_type="diag", reg_covar=1e-04).fit(z)
                # get GMM parameters
                phi = np.log(gmm.weights_ + self.eps) # inverse softmax
                mu = gmm.means_
                def inverse_softplus(x):
                    b = x < 1e2
                    x[b] = np.log(np.exp(x[b]) - 1.0 + self.eps)
                    return x
                sigma2 = inverse_softplus(gmm.covariances_)

                # initialize mixture components
                sess, saver, graph = self._restore_session()
                def my_get_variable(varname):
                    return [v for v in tf.global_variables() if v.name == varname][0]
                mu_c_unscaled = my_get_variable("mu_c_unscaled:0")
                sess.run(mu_c_unscaled.assign(tf.convert_to_tensor(mu, dtype=tf.float32)))
                sigma2_c_unscaled = my_get_variable("sigma2_c_unscaled:0")
                sess.run(sigma2_c_unscaled.assign(tf.convert_to_tensor(sigma2, dtype=tf.float32)))
                phi_c_unscaled = my_get_variable("phi_c_unscaled:0")
                sess.run(phi_c_unscaled.assign(tf.convert_to_tensor(phi, dtype=tf.float32)))
                saver.save(sess, self.save_path)
                sess.close()
            except:
                warnings.warn("Failed to initialize VaDER with Gaussian mixture")
            finally:
                pass
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
        G_c = np.ones(X_c.shape)
        sess, saver, graph = self._restore_session()
        z = graph.get_tensor_by_name("z:0")
        mu_tilde = graph.get_tensor_by_name("mu_tilde:0")
        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        G = graph.get_tensor_by_name("G_input:0")
        latent = np.concatenate([z.eval(feed_dict={X: X_c, W: W_c, G: G_c}) for i in np.arange(n_samp)], axis=0)
        sess.close()
        return latent

    def get_loss(self, X_c, W_c=None, mu_c=None, sigma2_c=None, phi_c=None):
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
            sigma2_c : float
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
        G = graph.get_tensor_by_name("G_input:0")
        alpha_t = graph.get_tensor_by_name("alpha_input:0")

        def my_get_variable(varname):
            return [v for v in tf.global_variables() if v.name == varname][0]

        mu_c_unscaled = my_get_variable("mu_c_unscaled:0")
        if mu_c is not None:
            mu_c_new = mu_c
            mu_c_old = mu_c_unscaled.eval()
            sess.run(mu_c_unscaled.assign(tf.convert_to_tensor(mu_c_new, dtype=tf.float32)))

        sigma2_c_unscaled = my_get_variable("sigma2_c_unscaled:0")
        if sigma2_c is not None:
            sigma2_c_new = sigma2_c
            sigma2_c_old = sigma2_c_unscaled.eval()
            sess.run(sigma2_c_unscaled.assign(tf.contrib.distributions.softplus_inverse(tf.convert_to_tensor(sigma2_c_new, dtype=tf.float32))))

        phi_c_unscaled = my_get_variable("phi_c_unscaled:0")
        if phi_c is not None:
            phi_c_new = phi_c
            phi_c_old = phi_c_unscaled.eval()
            sess.run(phi_c_unscaled.assign(tf.convert_to_tensor(phi_c_new, dtype=tf.float32)))

        G_c = 1 / np.bincount(self.groups)[self.groups]
        G_c = G_c / sum(G_c)
        G_c = np.broadcast_to(G_c, X_c.shape)

        lat_loss = lat_loss.eval(feed_dict={X: X_c, W: W_c, G: G_c, alpha_t: self.alpha})
        rec_loss = rec_loss.eval(feed_dict={X: X_c, W: W_c, G: G_c, alpha_t: self.alpha})

        sess.close()
        return {"reconstruction_loss": rec_loss, "latent_loss": lat_loss}

    def get_imputation_matrix(self):
        '''
            Returns
            -------
            The imputation matrix.
        '''
        sess, saver, graph = self._restore_session()
        def my_get_variable(varname):
            return [v for v in tf.global_variables() if v.name == varname][0]
        A = my_get_variable("A:0").eval()
        sess.close()
        return A

    def cluster(self, X_c, W_c=None, mu_c=None, sigma2_c=None, phi_c=None):
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
            sigma2_c : float
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
        G = graph.get_tensor_by_name("G_input:0")
        if mu_c is None:
            mu_c = graph.get_tensor_by_name("mu_c:0").eval()
        if sigma2_c is None:
            sigma2_c = graph.get_tensor_by_name("sigma2_c:0").eval()
        if phi_c is None:
            phi_c = graph.get_tensor_by_name("phi_c:0").eval()

        if W_c is None:
            W_c = np.ones(X_c.shape)
        G_c = np.ones(X_c.shape)

        clusters = self._cluster(mu_tilde.eval(feed_dict={X: X_c, W: W_c, G: G_c}), mu_c, sigma2_c, phi_c)
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
            Dictionary with three components, "mu", "sigma2_sq", "phi".
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()
        sigma2_c = graph.get_tensor_by_name("sigma2_c:0")
        mu_c = graph.get_tensor_by_name("mu_c:0")
        phi_c = graph.get_tensor_by_name("phi_c:0")

        res = {
            "mu": mu_c.eval(),
            "sigma2_sq": sigma2_c.eval(),
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
        sigma2_c = graph.get_tensor_by_name("sigma2_c:0")
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
                np.sqrt(sigma2_c.eval())[None, k, :] * \
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

        G_test = np.ones(X_test.shape)

        if self.seed is not None:
            np.random.seed(self.seed)

        sess, saver, graph = self._restore_session()

        X = graph.get_tensor_by_name("X_input:0")
        W = graph.get_tensor_by_name("W_input:0")
        G = graph.get_tensor_by_name("G_input:0")
        x_output = graph.get_tensor_by_name("x_output:0")

        pred = x_output.eval(feed_dict={X: X_test, W: W_test, G: G_test})
        sess.close()
        return pred
