#################################
# Your name: Ran Tuchman
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from sklearn import svm
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings("ignore")  # ignore Warnings: living on the edge


"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_clf = svm.SVC(C=1000.0, kernel='linear')
    quadratic_clf = svm.SVC(C=1000.0, kernel='poly', degree=2)
    rbf_clf = svm.SVC(C=1000.0, kernel='rbf')
    linear_clf.fit(X_train, y_train)
    quadratic_clf.fit(X_train, y_train)
    rbf_clf.fit(X_train, y_train)

    ##only for saving decision boundaries
    #clf_list = (linear_clf, quadratic_clf, rbf_clf)
    #for clf in clf_list:
    #    create_plot(X_train, y_train, clf)
    #    plt.title('Decision boundaries for {} kernel'.format(clf.kernel))
    #    plt.savefig(r"C:\Users\rtuchman\PycharmProjects\Intro-to-Machine-Learning\ex4\Decision boundaries for differnt kernels\Q_1a_{}.png".format(clf.kernel))
    #    plt.close()

    return np.vstack((linear_clf.n_support_, quadratic_clf.n_support_, rbf_clf.n_support_))


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    clf_list = []
    C_list = [10**x for x in range(-5, 6)]
    for c in C_list:
        linear_clf = svm.SVC(C=c, kernel='linear')
        linear_clf.fit(X_train, y_train)
        clf_list.append(linear_clf)
    accuracy_list = np.zeros(len(clf_list))
    for j in range(len(clf_list)):
        accuracy_list[j] = sum(clf_list[j].predict(X_val) == y_val) / float(len(X_val))

    plt.title('Accuracy as function of C')
    plt.plot(np.log10(C_list), accuracy_list)
    plt.grid()
    plt.xlabel('Log10(penalty constant)')
    plt.ylabel('Accuracy')
    plt.savefig('Q_1b.png')
    plt.close()

    ##only for saving decision boundaries#
    #for clf in clf_list:
    #    create_plot(X_train, y_train, clf)
    #    plt.title("Decision boundaries for C={}".format(clf.C))
    #    plt.savefig(r"C:\Users\rtuchman\PycharmProjects\Intro-to-Machine-Learning\ex4\Decision boundaries for C\Q_1b_C={}.png".format(clf.C))
    #plt.close()
    #return accuracy_list


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    clf_list = []
    gamma_list = [10**x for x in range(-5, 6)]
    for g in gamma_list:
        rbf_clf = svm.SVC(C=10, kernel='rbf', gamma=g)
        rbf_clf.fit(X_train, y_train)
        clf_list.append(rbf_clf)
    accuracy_list = np.zeros(len(clf_list))
    for j in range(len(clf_list)):
        accuracy_list[j] = sum(clf_list[j].predict(X_val) == y_val) / float(len(X_val))

    plt.title('Accuracy as function of gamma')
    plt.plot(np.log10(gamma_list), accuracy_list)
    plt.grid()
    plt.xlabel('Log10(gamma)')
    plt.ylabel('Accuracy')#
    plt.savefig('Q_1c.png')
    plt.close()

    ##only for saving decision boundaries
    #for clf in clf_list:
    #    create_plot(X_train, y_train, clf)
    #    plt.title('Decision boundaries for gamma={}'.format(clf.gamma))
    #    plt.savefig(r"C:\Users\rtuchman\PycharmProjects\Intro-to-Machine-Learning\ex4\Decision boundaries for gamma\Q_1c_gamma={}.png".format(clf.gamma))
    #plt.close()
    #return accuracy_list


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = get_points()
    train_three_kernels(X_train, y_train, X_val, y_val)
    linear_accuracy_per_C(X_train, y_train, X_val, y_val)
    rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)
