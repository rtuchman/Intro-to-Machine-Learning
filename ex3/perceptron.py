#################################
# Your name: Ran Tuchman
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import sgd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
plt.ioff()
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarning: Function mldata_filename is deprecated


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels):
    """
	returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    w = np.zeros(shape=(data.shape[1],))
    data = sklearn.preprocessing.normalize(data)
    for j in range(len(data)):
        if sign(np.dot(w, data[j])) != labels[j]:
            w = np.add(np.array(w), np.array(data[j]) * labels[j])
    return w


def q_1a(train_data, train_labels, test_data, test_labels):
    n_list = [5, 10, 50, 100, 500, 1000, 5000]
    accuracy_list = np.zeros(shape=(len(n_list), 100))
    mean_accuracy = np.zeros(shape=len(n_list))
    percentile_5_list = np.zeros(shape=len(n_list))
    percentile_95_list = np.zeros(shape=len(n_list))
    num_inputs = range(len(train_data))
    for i in range(len(n_list)):
        for j in range(100):
            permuted_index = np.random.permutation(num_inputs)
            w = perceptron(train_data[permuted_index[:n_list[i]]], train_labels[permuted_index[:n_list[i]]])
            accuracy = 0
            for k in range(len(test_data)):
                y_tilda = sign(np.dot(w, test_data[k]))
                if y_tilda == test_labels[k]:
                    accuracy += 1
            accuracy_list[i][j] = float(accuracy) / len(test_data)
        mean_accuracy[i] = (np.mean(accuracy_list[i][:])).round(3)
        percentile_5_list[i] = np.percentile(accuracy_list[i][:], 5).round(3)
        percentile_95_list[i] = np.percentile(accuracy_list[i][:], 95).round(3)

    cells = [percentile_5_list, percentile_95_list, mean_accuracy]
    cols = ['%d' %i for i in n_list]
    rows = ['5%', '95%', 'mean']
    plt.axis('off')
    plt.axis('tight')
    plt.table(cellText=cells, rowLabels=rows, colLabels=cols, loc='center')
    plt.tight_layout()
    plt.savefig('Q_1a.png')
    plt.close()
    return


def q_1b(data, labels):
    w = perceptron(data, labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.savefig('Q_1b.png')
    plt.close()



def q_1c(train_data, train_labels, test_data, test_labels):
    w = perceptron(train_data, train_labels)
    accuracy = 0
    for j in range(len(test_data)):
        accuracy += (sign(np.dot(w, test_data[j])) == test_labels[j])
    print("\nThe accuracy of the classifier trained on the full training set, applied on the test set is: {} %\n"
          .format(float(accuracy / len(test_data)).__round__(5)))


def q_1d(train_data, train_labels, test_data, test_labels, test_data_unscaled):
    w = perceptron(train_data, train_labels)
    image_num = 0
    for j in range(len(test_data)):
        y_tilda = sign(np.dot(w, test_data[j]))
        if y_tilda != test_labels[j]:
            image_num += 1
            plt.imshow(np.reshape(test_data_unscaled[j], (28, 28)), interpolation='nearest')
            plt.savefig(("Q_1b_{}.png".format(image_num)))
        if image_num == 2:
            plt.close()
            return



def sign(x):
    return (1, -1)[x < 0]

if __name__ == "__main__":
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels, test_data_unscaled = sgd.helper()
    q_1a(train_data, train_labels, test_data, test_labels)
    q_1b(train_data, train_labels)
    q_1c(train_data, train_labels, test_data, test_labels)
    q_1d(train_data, train_labels, test_data, test_labels, test_data_unscaled)



