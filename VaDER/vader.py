import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics
from scipy.stats import multivariate_normal
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from vadermodel import VaderRNN, VaderFFN, VaderTransformer

class VADER:
    '''
        A VADER object represents a (recurrent) (variational) (Gaussian mixture) autoencoder
    '''
    def __init__(self, X_train, W_train=None, y_train=None, n_hidden=[12, 2], k=3, groups=None, output_activation=None,
        batch_size = 32, learning_rate=1e-3, alpha=1.0, phi=None, cell_type="LSTM", cell_params=None, recurrent=True,
        save_path=None, eps=1e-10, seed=None, n_thread=0):
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
                Cell type of the (recurrent) neural network in case len(self.X.shape) == 3. [LSTM, GRU, SimpleRNN]
                (default: "LSTM")
            cell_params : dict
                Dictionary with (key, value) pairs for cell_type-specific hyperparameters. Only used for
                cell_type == "Transformer".
                (defaults to: {'d_model': 8, 'num_layers': 2, 'num_heads': 2, 'dff': 32, 'max_pe': 1e2, 'rate': 0.0},
                interpreted as in https://www.tensorflow.org/tutorials/text/transformer#scaled_dot_product_attention.
                Note that the use of dropout can be debated, due to the regularizing properties of the variational
                layer.)
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
            cell_type : str
                Cell type of the recurrent neural network.
            cell_params : dict
                Dictionary with (key, value) pairs for cell_type-specific hyperparameters.
            recurrent : bool
                Train a recurrent autoencoder, or a non-recurrent autoencoder?
            save_path : str
                Location to save the Tensorflow model.
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
            accuracy : float
                If y_train is not none the current accuracy of this VADER object.
            cluster_purity : float
                If y_train is not none the current cluster purity of this VADER object.
            gmm : dict
                The current GMM in the latent space of the VaDER model.
            D : int
                self.X.shape[1]. The number of time points if self.recurrent is True, otherwise the number of variables.
            G : float
                Groups weights proportional to group sizes as specified in groups attribute.
            I : integer
                X_train.shape[2]. The number of variables if self.recurrent is True, otherwise not defined.
            model :
                The VaDER model
            optimizer :
                The optimizer used for training the model
        '''

        if seed is not None:
            np.random.seed(seed)

        # not guaranteed to work with anything else but 32-bit
        self.float_type = "float32"
        self.int_type = "int32"

        self.D = np.array(X_train.shape[1], dtype=self.int_type)  # dimensionality of input/output
        self.X = X_train.astype(self.float_type)
        if W_train is not None:
            self.W = W_train.astype(self.float_type)
        else:
            self.W = np.ones(X_train.shape, self.float_type)
        if y_train is not None:
            self.y = np.array(y_train, self.int_type)
        else:
            self.y = None
        self.save_path = save_path
        self.eps = eps
        self.alpha = alpha  # weight for the latent loss (alpha times the reconstruction loss weight)
        self.learning_rate = learning_rate
        self.K = k  # 10 number of mixture components (clusters)
        if groups is not None:
            self.groups = np.array(groups, self.float_type)
            self.G = 1.0 / np.bincount(groups).astype(self.float_type)[groups]
            self.G = self.G / sum(self.G)
            self.G = np.broadcast_to(self.G, self.X.shape)
        else:
            self.groups = np.ones(X_train.shape[-1], self.int_type)
            self.G = np.ones(X_train.shape, dtype=self.float_type)

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
        self.accuracy = np.array([])
        self.cluster_purity = np.array([])
        self.gmm = {'mu': None, 'sigma2': None, 'phi': None}
        self.n_param = None
        self.cell_type = cell_type
        self.cell_params = cell_params
        if cell_type == "Transformer" and cell_params is None:
            self.cell_params = {'d_model': 8, 'num_layers': 2, 'num_heads': 2, 'dff': 32, 'rate': 0.1}
        self.recurrent = recurrent
        # experiment: encode as np.array
        if self.recurrent:
            self.I = np.array(X_train.shape[2], dtype=self.int_type)  # multivariate dimensions
        else:
            self.I = np.array(1, dtype=self.int_type)

        if self.seed is not None:
            tf.random.set_seed(self.seed)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, name="optimizer", clipnorm=5.0)

        if not self.recurrent:
            self.model = VaderFFN(
                self.X, self.W, self.D, self.K, self.I, self.cell_type, self.n_hidden, self.recurrent,
                self.output_activation, self.cell_params)
        elif self.cell_type == "Transformer":
            self.model = VaderTransformer(
                self.X, self.W, self.D, self.K, self.I, self.cell_type, self.n_hidden, self.recurrent,
                self.output_activation, self.cell_params)
        else:
            self.model = VaderRNN(
                self.X, self.W, self.D, self.K, self.I, self.cell_type, self.n_hidden, self.recurrent,
                self.output_activation, self.cell_params)

        # the state of the untrained model
        self._update_state(self.model)
        self.n_param = np.sum([np.product([xi for xi in x.shape]) for x in self.model.trainable_variables])

        if self.save_path is not None:
            tf.keras.models.save_model(self.model, self.save_path, save_format="tf")

    def fit(self, n_epoch=10, learning_rate=None, verbose=False, exclude_variables=None):
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
            exclude_variables: list of character
                List of variables to exclude in computing the gradient (currently not used)

            Returns
            -------
            0 if successful
        '''

        @tf.function
        def train_step(X, W, G):
            with tf.GradientTape() as tape:
                x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde = self.model((X, W), training=True)
                rec_loss = self._reconstruction_loss(
                    X, x, x_raw, G * W, self.output_activation, self.D, self.I, self.eps)
                if self.alpha > 0.0:
                    lat_loss = self.alpha * self._latent_loss(
                        z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, self.K, self.eps)
                else:
                    lat_loss = tf.convert_to_tensor(value=0.0, dtype=self.float_type)  # non-variational
                loss = rec_loss + lat_loss
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.seed is not None:
            np.random.seed(self.seed)

        if verbose:
            self._print_progress(self.model, -1)
        for epoch in range(n_epoch):
            n_batches = self.X.shape[0] // self.batch_size
            for iteration in range(n_batches):
                X_batch, y_batch, W_batch, G_batch = self._get_batch(self.batch_size)
                train_step(X_batch, W_batch, G_batch)
            self._update_state(self.model)
            self.n_epoch += 1
            if verbose:
                self._print_progress(self.model, self.n_epoch)
        if self.save_path is not None:
            tf.keras.models.save_model(self.model, self.save_path, save_format="tf")
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
                def my_get_variable(varname):
                    return [v for v in self.model.trainable_variables if v.name == varname][0]
                mu_c_unscaled = my_get_variable("mu_c_unscaled:0")
                mu_c_unscaled.assign(tf.convert_to_tensor(value=mu, dtype=self.float_type))
                sigma2_c_unscaled = my_get_variable("sigma2_c_unscaled:0")
                sigma2_c_unscaled.assign(tf.convert_to_tensor(value=sigma2, dtype=self.float_type))
                phi_c_unscaled = my_get_variable("phi_c_unscaled:0")
                phi_c_unscaled.assign(tf.convert_to_tensor(value=phi, dtype=self.float_type))
                self.gmm = {'mu': mu, 'sigma2': sigma2, 'phi': phi}
            except:
                warnings.warn("Failed to initialize VaDER with Gaussian mixture")
            finally:
                pass
        # restore the alpha
        self.alpha = alpha
        return ret

    # @tf.function
    def _update_state(self, model):
        rec_loss = lat_loss = acc = pur = 0
        n_max = 10
        for _ in np.arange(n_max):
            X_batch, y_batch, W_batch, G_batch = self._get_batch(self.batch_size)
            x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde = model((X_batch, W_batch))
            rec_loss = rec_loss + self._reconstruction_loss(
                X_batch, x, x_raw, G_batch * W_batch, self.output_activation, self.D, self.I,
                self.eps)
            lat_loss = lat_loss + self._latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, self.K, self.eps)
            loss = rec_loss + self.alpha * lat_loss
            if y_batch is not None:
                clusters = self._cluster(mu_tilde, mu_c, sigma2_c, phi_c)
                acc = acc + self._accuracy(clusters, y_batch)
                pur = pur + self._cluster_purity(clusters, y_batch)
        rec_loss /= n_max
        lat_loss /= n_max
        loss /= n_max
        acc /= n_max
        pur /= n_max

        # X_batch, y_batch, W_batch, G_batch = self._get_batch(min(10 * self.batch_size, self.X.shape[0]))
        # x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde = model((X_batch, W_batch))
        # rec_loss = self._reconstruction_loss(
        #     X_batch, x, x_raw, (G_batch * W_batch).astype(self.float_type), self.output_activation, self.D, self.I, self.eps)
        # lat_loss = self._latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, self.K, self.eps)
        # loss = rec_loss + self.alpha * lat_loss

        self.reconstruction_loss = np.append(self.reconstruction_loss, rec_loss)
        self.latent_loss = np.append(self.latent_loss, lat_loss)
        self.loss = np.append(self.loss, loss)
        self.gmm['mu'] = mu_c.numpy()
        self.gmm['sigma2'] = sigma2_c.numpy()
        self.gmm['phi'] = phi_c.numpy()
        if y_batch is not None:
            self.accuracy = np.append(self.accuracy, acc)
            self.cluster_purity = np.append(self.cluster_purity, pur)

    @tf.function
    def _reconstruction_loss(self, X, x, x_raw, W, output_activation, D, I, eps=1e-10):
        # reconstruction loss: E[log p(x|z)]
        if (output_activation == tf.nn.sigmoid):
            rec_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.clip_by_value(X, eps, 1 - eps), x_raw, W)
        else:
            rec_loss = tf.compat.v1.losses.mean_squared_error(X, x, W)
        rec_loss = tf.cast(rec_loss, self.float_type)

        # re-scale the loss to the original dims (making sure it balances correctly with the latent loss)
        num = tf.cast(tf.reduce_prod(input_tensor=tf.shape(input=W)), self.float_type)
        den = tf.cast(tf.reduce_sum(input_tensor=W), self.float_type)
        rec_loss = rec_loss * num / den
        rec_loss = rec_loss * self.D * self.I

        return rec_loss

    @tf.function
    def _latent_loss(self, z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, K, eps=1e-10):
        sigma2_tilde = tf.math.exp(log_sigma2_tilde)
        log_sigma2_c = tf.math.log(eps + sigma2_c)
        if K == 1:  # ordinary VAE
            latent_loss = tf.reduce_mean(input_tensor=0.5 * tf.reduce_sum(
                input_tensor=sigma2_tilde + tf.square(mu_tilde) - 1 - log_sigma2_tilde,
                axis=1
            ))
        else:
            log_2pi = tf.math.log(2 * np.pi)
            log_phi_c = tf.math.log(eps + phi_c)

            # def f(i):
            #     return - 0.5 * (log_sigma2_c[i] + log_2pi + tf.math.square(z - mu_c[i]) / sigma2_c[i])
            # log_pdf_z = tf.transpose(a=tf.map_fn(f, np.arange(K), fn_output_signature=self.float_type), perm=[1, 0, 2])

            N = z.shape[0]
            ii, jj = tf.meshgrid(tf.range(K, dtype = self.int_type), tf.range(N, dtype = self.int_type))
            ii = tf.reshape(ii, [N * K])
            jj = tf.reshape(jj, [N * K])
            lsc_b = tf.gather(log_sigma2_c, ii, axis = 0)
            mc_b = tf.gather(mu_c, ii, axis = 0)
            sc_b = tf.gather(sigma2_c, ii, axis = 0)
            z_b = tf.gather(z, jj, axis = 0)
            log_pdf_z = - 0.5 * (lsc_b + log_2pi + tf.math.square(z_b - mc_b) / sc_b)
            log_pdf_z = tf.reshape(log_pdf_z, [N, K, self.n_hidden[-1]])

            log_p = log_phi_c + tf.reduce_sum(input_tensor=log_pdf_z, axis=2)
            lse_p = tf.reduce_logsumexp(input_tensor=log_p, keepdims=True, axis=1)
            log_gamma_c = log_p - lse_p

            gamma_c = tf.exp(log_gamma_c)

            # # latent loss: E[log p(z|c) + log p(c) - log q(z|x) - log q(c|x)]
            # term1 = tf.math.log(eps + sigma2_c)
            # f2 = lambda i: sigma2_tilde / (eps + sigma2_c[i])
            # term2 = tf.transpose(a=tf.map_fn(f2, np.arange(K), fn_output_signature=self.float_type), perm=[1, 0, 2])
            # f3 = lambda i: tf.square(mu_tilde - mu_c[i]) / (eps + sigma2_c[i])
            # term3 = tf.transpose(a=tf.map_fn(f3, np.arange(K), fn_output_signature=self.float_type), perm=[1, 0, 2])

            # latent loss: E[log p(z|c) + log p(c) - log q(z|x) - log q(c|x)]
            term1 = tf.math.log(eps + sigma2_c)
            N = sigma2_tilde.shape[0]
            ii, jj = tf.meshgrid(tf.range(K, dtype = self.int_type), tf.range(N, dtype = self.int_type))
            ii = tf.reshape(ii, [N * K])
            jj = tf.reshape(jj, [N * K])
            st_b = tf.gather(sigma2_tilde, jj, axis = 0)
            sc_b = tf.gather(sigma2_c, ii, axis = 0)
            term2 = tf.reshape(st_b / (eps + sc_b), [N, K, self.n_hidden[-1]])
            mt_b = tf.gather(mu_tilde, jj, axis = 0)
            mc_b = tf.gather(mu_c, ii, axis = 0)
            term3 = tf.reshape(tf.math.square(mt_b - mc_b) / (eps + sc_b), [N, K, self.n_hidden[-1]])

            latent_loss1 = 0.5 * tf.reduce_sum(
                input_tensor=gamma_c * tf.reduce_sum(input_tensor=term1 + term2 + term3, axis=2), axis=1)
            # latent_loss2 = - tf.reduce_sum(gamma_c * tf.log(eps + phi_c / (eps + gamma_c)), axis=1)
            latent_loss2 = - tf.reduce_sum(input_tensor=gamma_c * (log_phi_c - log_gamma_c), axis=1)
            latent_loss3 = - 0.5 * tf.reduce_sum(input_tensor=1 + log_sigma2_tilde, axis=1)
            # average across the samples
            latent_loss1 = tf.reduce_mean(input_tensor=latent_loss1)
            latent_loss2 = tf.reduce_mean(input_tensor=latent_loss2)
            latent_loss3 = tf.reduce_mean(input_tensor=latent_loss3)
            # add the different terms
            latent_loss = latent_loss1 + latent_loss2 + latent_loss3
        return tf.cast(latent_loss, self.float_type)

    def _cluster(self, mu_t, mu, sigma2, phi):
        def f(mu_t, mu, sigma2, phi):
            # the covariance matrix is diagonal, so we can just take the product
            return np.log(self.eps + phi) + np.log(self.eps + multivariate_normal.pdf(mu_t, mean=mu, cov=np.diag(sigma2)))
        p = np.array([f(mu_t, mu[i], sigma2[i], phi[i]) for i in np.arange(mu.shape[0])])
        return np.argmax(p, axis=0)

    def _cluster_purity(self, y_pred, y_true):
        tab = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(tab, axis=0)) / np.sum(tab)

    def _accuracy(self, y_pred, y_true):
        def cluster_acc(Y_pred, Y):
            assert Y_pred.size == Y.size
            D = max(Y_pred.max(), Y.max()) + 1
            w = np.zeros((D, D), dtype=self.int_type)
            for i in range(Y_pred.size):
                w[Y_pred[i], Y[i]] += 1
            ind = np.transpose(np.asarray(linear_assignment(w.max() - w)))
            return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, np.array(w)

        y_pred = np.array(y_pred, self.int_type)
        y_true = np.array(y_true, self.int_type)
        return cluster_acc(y_pred, y_true)[0]

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

    def _print_progress(self, model, epoch=-1):
        if self.y is not None:
            print(epoch,
                  "tot_loss:", "%.2f" % round(self.loss[-1], 2),
                  "\trec_loss:", "%.2f" % round(self.reconstruction_loss[-1], 2),
                  "\tlat_loss:", "%.2f" % round(self.latent_loss[-1], 2),
                  "\tacc:", "%.2f" % round(self.accuracy[-1], 2),
                  "\tpur:", "%.2f" % round(self.cluster_purity[-1], 2),
                  flush=True
                  )
        else:
            print(epoch,
                  "tot_loss:", "%.2f" % round(self.loss[-1], 2),
                  "\trec_loss:", "%.2f" % round(self.reconstruction_loss[-1], 2),
                  "\tlat_loss:", "%.2f" % round(self.latent_loss[-1], 2),
                  flush=True
                  )
        return 0

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
            W_c = np.ones(X_c.shape, dtype=self.float_type)
        else:
            W_c = W_c.astype(X_c.dtype)
        return np.concatenate([self.model((X_c, W_c))[5] for i in np.arange(n_samp)], axis=0)

    def get_loss(self, X_c, W_c=None, mu_c=None, sigma2_c=None, phi_c=None):
        '''
            Calculate the loss for specific input data.

            Parameters
            ----------
            X_c : float
                The data to be clustered. Numpy array with dimensions [samples, time points, variables] if recurrent is
                True, else [samples, variables]
            W_c : int
                Missingness indicator. Numpy array with same dimensions as X_c. Entries in X_c for which the
                corresponding entry in W_train equals 0 are treated as missing. More precisely, their specific numeric
                value is completely ignored. If None, then no missingness is assumed. (default: None)

            Returns
            -------
            Dictionary with two components, "reconstruction_loss" and "latent_loss".
        '''
        if W_c is None:
            W_c = np.ones(X_c.shape, dtype=self.float_type)

        x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde = self.model((X_c, W_c))

        G_c = 1 / np.bincount(self.groups)[self.groups]
        G_c = G_c / sum(G_c)
        G_c = np.broadcast_to(G_c, X_c.shape)

        reconstruction_loss_val = self._reconstruction_loss(
            X_c, x, x_raw, G_c * W_c, self.output_activation, self.D, self.I, self.eps).numpy
        latent_loss_val = self._latent_loss(z, mu_c, sigma2_c, phi_c, mu_tilde, log_sigma2_tilde, self.K, self.eps).numpy
        return {"reconstruction_loss": reconstruction_loss_val, "latent_loss": latent_loss_val}

    def get_imputation_matrix(self):
        '''
            Returns
            -------
            The imputation matrix.
        '''
        return self.model.imputation_layer.A

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

            Returns
            -------
            Clusters encoded as integers.
        '''
        if W_c is None:
            W_c = np.ones(X_c.shape, self.float_type)
        else:
            W_c = W_c.astype(self.float_type)
        x, x_raw, mu_c, sigma2_c, phi_c, z, mu_tilde, log_sigma2_tilde = self.model((X_c, W_c))

        return self._cluster(mu_tilde, mu_c, sigma2_c, phi_c)

    def get_cluster_means(self):
        '''
            Get the cluster averages represented by this VADER object. Technically, this maps the latent Gaussian
            mixture means to output values using this VADER object.

            Returns
            -------
            Cluster averages.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        clusters = self.model.decode(self.gmm['mu_c'])
        return clusters.numpy()

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

        if self.recurrent:
            gen = np.zeros((n, self.D, self.I), dtype=np.float)
        else:
            gen = np.zeros((n, self.D), dtype=np.float)
        c = np.random.choice(np.arange(self.K), size=n, p=self.gmm['phi'])
        for k in np.arange(self.K):
            ii = np.flatnonzero(c == k)
            z_rnd = \
                self.gmm['mu'][None, k, :] + \
                np.sqrt(self.gmm['sigma2'])[None, k, :] * \
                np.random.normal(size=[ii.shape[0], self.n_hidden[-1]])
            if self.recurrent:
                gen[ii,:,:] = self.model.decode(z_rnd)[0].numpy()
            else:
                gen[ii,:] = self.model.decode(z_rnd)[0].numpy()
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
            W_test = np.ones(X_test.shape, self.float_type)
        else:
            W_test = W_test.astype(self.float_type)

        if self.seed is not None:
            np.random.seed(self.seed)

        return self.model((X_test, W_test))[0].numpy()

