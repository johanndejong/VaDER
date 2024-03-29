This repository contains code for a method for clustering multivariate time series with potentially many missing values (published [here](https://academic.oup.com/gigascience/article/8/11/giz134/5626377)), a setting commonly encountered in the analysis of longitudinal clinical data, but generally still poorly addressed in the literature. The method is based on a variational autoencoder with a Gaussian mixture prior (with a latent loss as described [here](https://arxiv.org/abs/1611.05148)), extended with LSTMs for modeling multivariate time series, as well as implicit imputation and loss re-weighting for directly dealing with (potentially many) missing values.

The use of the method is not restricted to clinical data. It can generally be used for multivariate time series data. 

In addition to variational autoencoders with gaussian mixture priors, the code allows to train ordinary variational autoencoders (multivariate gaussian prior) and ordinary autoencoders (without prior).
