#################################
# Your name: Ran Tuchman
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import sklearn.preprocessing
plt.ioff()
import warnings
warnings.filterwarnings("ignore")

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
        i = np.random.randint(0, len(data), 1)[0]
        eta_t = eta_0 / (t+1)
        w = w - eta_t*subgrad_sgd_hinge_loss(w, data[i], labels[i], C)
    return w


def q_2a(data, labels, validation_data, validation_labels, T):
    C = 1
    average_accuracy_list = [0 for _ in range(10)]
    eta_0_list = [10**-5*(10**i) for i in range(10)]
    for k in range(10):
        average_accuracy_list[k] = average_accuracy(data, labels, validation_data, validation_labels, C, eta_0_list[k], T)

    log_10_eta_0 = [np.log10(x) for x in eta_0_list]
    plt.title('Average accuracy as function of eta_0')
    plt.plot(log_10_eta_0, average_accuracy_list)
    plt.grid()
    plt.xlabel('Log10(eta_0)')
    plt.ylabel('Accuracy')
    plt.savefig('Q_2a.png')
    plt.close()
    return


def q_2b(data, labels, validation_data, validation_labels, T):
    eta_0 = 1
    average_accuracy_list = [0 for _ in range(10)]
    c_list = [10**-5*(10**i) for i in range(10)]
    for k in range(10):
        average_accuracy_list[k] = average_accuracy(data, labels, validation_data, validation_labels, c_list[k], eta_0, T)

    log_10_c = [np.log10(x) for x in c_list]
    plt.title('Average accuracy as function of C')
    plt.plot(log_10_c, average_accuracy_list)
    plt.grid()
    plt.xlabel('Log10(C)')
    plt.ylabel('Accuracy')
    plt.savefig('Q_2b.png')
    plt.close()
    return


def q_2c():
    best_C = 10**(-4)


def q_2d():
    pass


def sign(x):
    return (1, -1)[x < 0]


def subgrad_sgd_hinge_loss(w, x_i, y_i, c):
    if 1 - y_i*np.dot(x_i, w) < 0:
        return w
    else:
        return -c*y_i*x_i + w


def average_accuracy(data, labels, validation_data, validation_labels, C, eta_0, T):
    w_list = []
    accuracy_list = [0 for _ in range(10)]
    for i in range(10):
        w_list.append(SGD(data, labels, C, eta_0, T))
        for j in range(len(validation_data)):
            accuracy_list[i] += \
                (sign(np.dot(w_list[i], validation_data[j])) == validation_labels[j]) / len(validation_data)

    return np.mean(accuracy_list)



if __name__ == "__main__":
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, test_data_unscaled = helper()
    #q_2a(train_data, train_labels, validation_data, validation_labels, 1000)
    q_2b(train_data, train_labels, validation_data, validation_labels, 1000)





