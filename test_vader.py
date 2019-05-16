import matplotlib.pyplot as plt
import numpy as np
import os
from vader import VADER

save_path = os.path.join('test_vader', 'vader.ckpt')

nt = int(8)
ns = int(1.5e3)
sigma = .5
mu1 = -2
mu2 = 2
a1 = np.random.normal(mu1, sigma, ns)
a2 = np.random.normal(mu2, sigma, ns)
X0 = np.outer(a1, 2 * np.append(np.arange(-nt/2, 0), 0.5 * np.arange(0, nt/2)[1:]))
X1 = np.outer(a2, 0.5 * np.append(np.arange(-nt/2, 0), 2 * np.arange(0, nt/2)[1:]))

# X0 = X0 + np.random.normal(loc=0, scale=.1, size=nt*ns).reshape(ns, nt)
# X1 = X1 + np.random.normal(loc=0, scale=.1, size=nt*ns).reshape(ns, nt)
X_train = np.concatenate((X0, X1), axis=0)
y_train = np.repeat([0, 1], ns)
ii = np.random.choice(2*ns, 2*ns, replace=False)
X_train = X_train[ii,:]
X_train = np.expand_dims(X_train, 2)
y_train = y_train[ii]
W_train = np.random.choice(2, X_train.shape)

vader = VADER(X_train, save_path, n_hidden=[12, 1], k=2, learning_rate=1e-3, y_train=y_train,
                output_activation=None, recurrent=True, alpha=1.0, batch_size=64, weights=W_train)
vader.pre_fit(n_epoch=10, verbose=True)
vader.fit(n_epoch=10, verbose=True)
mu = vader.map_to_latent(X_train)
c = vader.cluster(X_train)
p = vader.predict(X_train)
g = vader.generate(100)

# plt.hist(mu)

for i in np.arange(100):
    plt.plot(g['samples'][i,])

# for i in np.arange(100):
#     plt.plot(p[i,])

