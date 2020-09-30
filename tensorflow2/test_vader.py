import numpy as np
import os
import sys
import tensorflow as tf
tf.config.run_functions_eagerly(False)
sys.path.append(os.getcwd())
sys.path.append('tensorflow2')
from vader import VADER

save_path = os.path.join('test_vader', 'vader.ckpt')

# generating some simple random data [ns * 2 samples, nt - 1 time points, 2 variables]
nt = int(8)
ns = int(2e2)
sigma = 0.5
mu1 = -2
mu2 = 2
a1 = np.random.normal(mu1, sigma, ns)
a2 = np.random.normal(mu2, sigma, ns)
# one variable with two clusters
X0 = np.outer(a1, 2 * np.append(np.arange(-nt/2, 0), 0.5 * np.arange(0, nt/2)[1:]))
X1 = np.outer(a2, 0.5 * np.append(np.arange(-nt/2, 0), 2 * np.arange(0, nt/2)[1:]))
X_train = np.concatenate((X0, X1), axis=0)
y_train = np.repeat([0, 1], ns)
# add another variable as a random permutation of the first one
# resulting in four clusters in total
ii = np.random.permutation(ns * 2)
X_train = np.stack((X_train, X_train[ii,]), axis=2)
# we now get four clusters in total
y_train = y_train * 2**0 + y_train[ii] * 2**1

# normalize (better for fitting)
for i in np.arange(X_train.shape[2]):
    X_train[:,:,i] = (X_train[:,:,i] - np.mean(X_train[:,:,i])) / np.std(X_train[:,:,i])

# randomly re-order the samples
ii = np.random.permutation(ns * 2)
X_train = X_train[ii,:]
y_train = y_train[ii]
# Randomly set 50% of values to missing (0: missing, 1: present)
# Note: All X_train[i,j] for which W_train[i,j] == 0 are treated as missing (i.e. their specific value is ignored)
W_train = np.random.choice(2, X_train.shape)

import pickle
pickle.dump(X_train, open("X_train.pickle", "wb"))
pickle.dump(y_train, open("y_train.pickle", "wb"))
pickle.dump(W_train, open("W_train.pickle", "wb"))

# Note: y_train is used purely for monitoring performance when a ground truth clustering is available.
# It can be omitted if no ground truth is available.
vader = VADER(X_train=X_train, W_train=W_train, y_train=y_train, save_path=save_path, n_hidden=[12, 2], k=4,
              learning_rate=1e-3, output_activation=None, recurrent=True, batch_size=16)

# pre-train without latent loss
vader.pre_fit(n_epoch=50, verbose=True)
# train with latent loss
vader.fit(n_epoch=50, verbose=True)
# get the clusters
c = vader.cluster(X_train)
# get the re-constructions
p = vader.predict(X_train)
# compute the loss given the network
l = vader.get_loss(X_train)

# Run VaDER non-recurrently (ordinary VAE with GM prior)
nt = int(8)
ns = int(2e2)
sigma = np.diag(np.repeat(2, nt))
mu1 = np.repeat(-1, nt)
mu2 = np.repeat(1, nt)
a1 = np.random.multivariate_normal(mu1, sigma, ns)
a2 = np.random.multivariate_normal(mu2, sigma, ns)
X_train = np.concatenate((a1, a2), axis=0)
y_train = np.repeat([0, 1], ns)
ii = np.random.permutation(ns * 2)
X_train = X_train[ii,:]
y_train = y_train[ii]
# normalize (better for fitting)
X_train = (X_train - np.mean(X_train)) / np.std(X_train)

vader = VADER(X_train=X_train, y_train=y_train, n_hidden=[12, 2], k=2, learning_rate=1e-3, output_activation=None,
              recurrent=False, batch_size=16)
# pre-train without latent loss
vader.pre_fit(n_epoch=50, verbose=True)
# train with latent loss
vader.fit(n_epoch=50, verbose=True)
# get the clusters
c = vader.cluster(X_train)
# get the re-constructions
p = vader.predict(X_train)
# compute the loss given the network
l = vader.get_loss(X_train)
# generate some samples
g = vader.generate(10)
