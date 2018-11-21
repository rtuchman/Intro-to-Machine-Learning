#################################
# Your name: Ran Tuchman
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

  """
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():

    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0, 8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

  # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels, test_data_unscaled

def SGD(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    w = np.zeros(shape=(data.shape[1],))
    for t in range(T):
        i = np.random.uniform(0, len(data), 1)
        eta_t = eta_0 / t
        w = w - eta_t*subgrad_sgd_hinge_loss(w, data[i], labels[i], C)
    return w





def subgrad_sgd_hinge_loss(w, x_i, y_i, c):
    if  1 - y_i*np.dot(w, x_i) < 0:
        return w
    else:
        return -c*y_i*x_i + w


