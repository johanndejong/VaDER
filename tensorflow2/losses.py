import tensorflow as tf
import numpy as np

def reconstruction_loss(X, x, x_raw, W, output_activation, D, I, eps=1e-10):
    # reconstruction loss: E[log p(x|z)]

    if (output_activation == tf.nn.sigmoid):
        rec_loss = tf.compat.v1.losses.sigmoid_cross_entropy(tf.clip_by_value(X, eps, 1 - eps), x_raw, W)
    else:
        rec_loss = tf.compat.v1.losses.mean_squared_error(X, x, W)

    # re-scale the loss to the original dims (making sure it balances correctly with the latent loss)
    rec_loss = rec_loss * tf.cast(tf.reduce_prod(input_tensor=tf.shape(input=W)), dtype=tf.float32) / tf.reduce_sum(input_tensor=W)
    rec_loss = D * I * rec_loss

    return rec_loss


def latent_loss(z, mu_c, sigma_c, phi_c, mu_tilde, log_sigma_tilde, K, eps=1e-10):
    if K == 1:  # ordinary VAE
        latent_loss = tf.reduce_mean(input_tensor=0.5 * tf.reduce_sum(
            input_tensor=tf.square(tf.exp(log_sigma_tilde)) + tf.square(mu_tilde) - 1 - 2 * log_sigma_tilde,
            axis=1
        ))
    else:
        # latent_loss = 0.5 * tf.reduce_sum(
        #     tf.square(tf.exp(log_sigma_tilde)) + tf.square(mu_tilde)
        #     - 1 - tf.log(eps + tf.square(tf.exp(log_sigma_tilde))))
        # latent_loss = tf.identity(latent_loss, name = "latent_loss")

        def log_pdf(z, mu, sigma):
            def f(i):
                return - tf.square(z - mu[i]) / 2.0 / (eps + sigma[i]) - tf.math.log(
                    eps + eps + 2.0 * np.pi * sigma[i]) / 2.0

            return tf.transpose(a=tf.map_fn(f, np.arange(K), dtype=tf.float32), perm=[1, 0, 2])

        # log_p = tf.reduce_sum(tf.log(phi_c) - 0.5 * tf.log(2 * np.pi * sigma_c))

        # log_p = tf.log(eps + phi_c) + tf.reduce_sum(log_pdf(z, mu_c, sigma_c), axis=2)
        # gamma_c = tf.nn.softmax(tf.exp(log_p))
        # log_gamma_c = tf.log(gamma_c + eps)

        log_p = tf.math.log(eps + phi_c) + tf.reduce_sum(input_tensor=log_pdf(z, mu_c, sigma_c), axis=2)
        lse_p = tf.reduce_logsumexp(input_tensor=log_p, keepdims=True, axis=1)
        log_gamma_c = log_p - lse_p

        gamma_c = tf.exp(log_gamma_c)

        # latent loss: E[log p(z|c) + log p(c) - log q(z|x) - log q(c|x)]
        term1 = tf.math.log(eps + sigma_c)
        term2 = tf.transpose(
            a=tf.map_fn(lambda i: tf.exp(log_sigma_tilde) / (eps + sigma_c[i]), np.arange(K), tf.float32),
            perm=[1, 0, 2])
        term3 = tf.transpose(
            a=tf.map_fn(lambda i: tf.square(mu_tilde - mu_c[i]) / (eps + sigma_c[i]), np.arange(K),
                      tf.float32), perm=[1, 0, 2])

        latent_loss1 = 0.5 * tf.reduce_sum(input_tensor=gamma_c * tf.reduce_sum(input_tensor=term1 + term2 + term3, axis=2), axis=1)
        # latent_loss2 = - tf.reduce_sum(gamma_c * tf.log(eps + phi_c / (eps + gamma_c)), axis=1)
        latent_loss2 = - tf.reduce_sum(input_tensor=gamma_c * (tf.math.log(eps + phi_c) - log_gamma_c), axis=1)
        latent_loss3 = - 0.5 * tf.reduce_sum(input_tensor=1 + log_sigma_tilde, axis=1)
        # average across the samples
        latent_loss1 = tf.reduce_mean(input_tensor=latent_loss1)
        latent_loss2 = tf.reduce_mean(input_tensor=latent_loss2)
        latent_loss3 = tf.reduce_mean(input_tensor=latent_loss3)
        # add the different terms
        latent_loss = latent_loss1 + latent_loss2 + latent_loss3
    return latent_loss

