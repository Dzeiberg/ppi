#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# default_exp nnpu


# # NNPU
# 
# > Implementations of [Non-Negative Risk Estimator](https://arxiv.org/abs/1703.00593) and [AbsNNPU](https://papers.nips.cc/paper/2020/file/98b297950041a42470269d56260243a1-Paper.pdf)

# In[ ]:


# export
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import losses
import tensorflow.keras.backend as K
import numpy as np
from scipy.stats import bernoulli
from easydict import EasyDict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 

import scipy.stats as ss

from sklearn.model_selection import train_test_split


# In[ ]:


# export
class Basic(tf.keras.Model):

    def __init__(self, n_units, n_hidden, dropout_rate):
        super(Basic, self).__init__()
        self.Dens = list()
        self.BN = list()
        self.Drop = list()
        for i in np.arange(n_hidden):
            if i == 0:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            else:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            self.BN.append(layers.BatchNormalization())
            self.Drop.append(layers.Dropout(dropout_rate))
        self.dens_last = layers.Dense(1)
        # self.BN_last = layers.BatchNormalization()
        # self.sigmoid = activations.sigmoid()

    def call(self, inputs):
        for i in np.arange(len(self.Dens)):
            if i == 0:
                x = self.Dens[i](inputs)
            else:
                x = self.Dens[i](x)
            x = self.BN[i](x)
            x = self.Drop[i](x)
        x = self.dens_last(x)
        # x = self.BN_last(x)
        return activations.sigmoid(x)


# In[ ]:


# export
def NNPULoss(alpha):
    epsilon = 10 ** -10

    def loss_function(y_true, pn_posterior):
        i_zero = K.flatten(tf.equal(y_true, 0))
        i_one = K.flatten(tf.equal(y_true, 1))
        pn_posterior_0 = tf.boolean_mask(pn_posterior[:, 0], i_zero, axis=0)
        pn_posterior_1 = tf.boolean_mask(pn_posterior[:, 0], i_one, axis=0)
        loss_neg = -tf.reduce_mean(tf.math.log(1 - pn_posterior_0 + epsilon))
        loss_neg = tf.maximum(0.0, loss_neg + alpha * tf.reduce_mean(tf.math.log(1 - pn_posterior_1 + epsilon)))
        loss_pos = -alpha * tf.reduce_mean(tf.math.log(pn_posterior_1 + epsilon))
        return loss_neg + loss_pos

    return loss_function


def NNPUAbsLoss(alpha):
    epsilon = 10 ** -10

    def loss_function(y_true, pn_posterior):
        i_zero = K.flatten(tf.equal(y_true, 0))
        i_one = K.flatten(tf.equal(y_true, 1))
        pn_posterior_0 = tf.boolean_mask(pn_posterior[:, 0], i_zero, axis=0)
        pn_posterior_1 = tf.boolean_mask(pn_posterior[:, 0], i_one, axis=0)
        loss_neg = -tf.reduce_mean(tf.math.log(1 - pn_posterior_0 + epsilon))
        loss_neg = tf.math.abs(loss_neg + alpha * tf.reduce_mean(tf.math.log(1 - pn_posterior_1 + epsilon)))
        loss_pos = -alpha * tf.reduce_mean(tf.math.log(pn_posterior_1 + epsilon))
        return loss_neg + loss_pos

    return loss_function


# In[ ]:


# export
def gradients(net, x, y, LossFnc):
    #YGen = np.cast['float32'](np.concatenate((y,pn_posterior_old, disc_posterior), axis=1))
    with tf.GradientTape() as tape:
        #pdb.set_trace()
        loss = LossFnc(y, net(x))
    return loss, tape.gradient(loss, net.trainable_variables)


# In[ ]:


# export
def batch(x, y, n_p, n_u):
    x_p, ix_p = batchPos(x, y, n_p)
    x_u, ix_u = batchUL(x, y, n_u)
    xx = np.concatenate((x_p, x_u), axis=0)
    ix = np.concatenate((ix_p, ix_u), axis=0)
    return xx, y[ix, :], x_p, x_u, ix


def batchPos(x, y, n_p):
    return batchY(x, y, 1, n_p)


def batchUL(x, y, n_u):
    return batchY(x, y, 0, n_u)

def batchY(x, y, value, n, *args):
    ix = (y == value).flatten( )
    ix_all = np.arange(np.size(y))
    ix = ix_all[ix]
    if args:
        p = args[0].flatten()
        p = p[ix]
        ix_p = bernoulli.rvs(p)
        ix_p = np.cast['bool'](ix_p)
        ix = ix[ix_p]
    ix = np.random.choice(ix, n, replace=True)
    xx = x[ix, :]
    return xx, ix


# In[ ]:


# export
def getPosterior(x,y,alpha,
                 inputs=None,
                 pupost=None,
                 training_args=EasyDict({"n_units":1000,
                                         "n_hidden":10,
                                         "dropout_rate":0.5,
                                         "maxIter":500,
                                         "batch_size":128}),
                 distributions=None,
                 viz_freq=10,
                 plotDistrs=False,
                absLoss=True,
                yPN=None):
    """
    x : (n x d) array
    y : (n x 1) array
    alpha : float
    training_args: EasyDict
        n_units : default 20 : size of hiddden layers
        n_hidden : default 10 : number of hidden layers
        dropout_rate : default 0.1 : drop percentage
        maxIter : default 100 : number of epochs
        batch_size : default 500 : batch size
    distributions : EasyDict :
        true_posterior(x) : callable
        f1(x) : callable
        f0(x) : callable
    viz_freq : default 10 : if distributions is specified, plot the 1D distributions at this period
    """
    # model
    net = Basic(training_args.n_units,
                training_args.n_hidden,
                training_args.dropout_rate)
    # loss
    if absLoss:
        LossFnc = NNPUAbsLoss(alpha)
    else:
        LossFnc = NNPULoss(alpha)
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if pupost is not None:
        inputs = pupost(x)
    elif inputs is not None:
        inputs = inputs
    else:
        inputs = x
    inputsTrain,inputsVal,labelsTrain,labelsVal = train_test_split(inputs,y)
    
    def plot():
        estimatedPosterior = net.predict(inputs)[:,0].ravel()
        truePosterior = distributions.true_posterior(x).ravel()
        plt.scatter(estimatedPosterior, truePosterior,alpha=.1,label="mae: {:.3f}".format(np.mean(np.abs(estimatedPosterior - truePosterior))))
        plt.plot([0,1],[0,1],color="black")
        plt.xlabel("estimated posterior")
        plt.ylabel("true posterior")
        plt.legend()
        plt.show()
    minLoss,patience = np.inf,0
    for i in tqdm(range(training_args.maxIter),total=training_args.maxIter, leave=False):
        xx,yy,_,_,ix = batch(inputsTrain,labelsTrain,training_args.batch_size,training_args.batch_size)
        loss, grads = gradients(net,xx,yy,LossFnc)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        valloss,_ = gradients(net,inputsVal, labelsVal,LossFnc)
        if valloss < minLoss:
            minLoss = valloss
            patience = 0
        else:
            patience += 1
        if distributions is not None and not i % viz_freq:
            plot()
        if patience == 50:
            break
    if distributions is not None:
        plot()
    return net.predict(inputs),net
