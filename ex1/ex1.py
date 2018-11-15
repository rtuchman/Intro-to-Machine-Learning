from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from heapq import *
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarning: Function mldata_filename is deprecated

# Student name: Ran Tuchman

"""This module computes the k-NN for a query image in the MNIST data set.

   To run from terminal simply type: python ex1.py

"""

# Some functions to help knn
def calc_distances(query, data, index):
    """go through all images and calc distance, store result in a sorted list"""

    distances_heap = []
    for i in range(len(data)):
        heappush(distances_heap, (distance.euclidean(query, data[i]), index + i))
    return distances_heap


def find_closest_neighbor_in_k_neighbors(labels, distances_heap, k):
    """calc histogram of k neighbors and find the one label with most instances in k closest neighbors"""
    k_labels_dict = {}
    for i in range(k):
        closest = heappop(distances_heap)
        label = labels[closest[1]]
        k_labels_dict[label] = k_labels_dict.get(label, 0) + (1/closest[0])**2  # use (1/distance)^2 to improve accuracy
    #find label woth most instances
    max_label = 0
    max_count = 0

    for key, val in k_labels_dict.items():
        if max_count < val:
            max_count = val
            max_label = key

    return max_label

def knn(data, labels, query, k):
    """finds k-Nearest Neighbors of a query image in a data set"""
    distances_heap = calc_distances(query, data, 0)
    return find_closest_neighbor_in_k_neighbors(labels, distances_heap, k)


def q_b(train_data, train_labels, test_data, test_labels, n=1000, k=10, print_result=True):
    error = 0.0

    for i in range(1000):
        prediction = knn(train_data[:n], train_labels[:n], test_data[i], k)
        error += (prediction != test_labels[i])
    accuracy = 1 - error / n

    print print_result*("\nThe accuracy of the k-NN algorithm with k=10 is: {}\n".format(accuracy))
    return accuracy


def q_c(train_data, train_labels, test_data, test_labels):
    accuracy_lst = []

    for j in range(1, 101):
        accuracy_lst.append(q_b(train_data, train_labels, test_data, test_labels, k=j, print_result=False))
        print "done computing {}%".format(j)*(j % 10 == 0)

    print "done computing accuracy list for q_1c"
    plt.title('Accuracy as a function of k - nearest neighbors')
    plt.plot([k + 1 for k in range(100)], accuracy_lst, '.')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('q_1c.png')


def q_d(train_data, train_labels, test_data, test_labels):
    accuracy_lst = []

    for j in range(100, 5000, 100):
        accuracy_lst.append(q_b(train_data, train_labels, test_data, test_labels, n=j, k=1, print_result=False))
        print "done computing {}%".format((j*100)/5000)*(((j*100)/5000) % 10 == 0)

    plt.title('Accuracy as a function of n - training images')
    plt.plot([n for n in range(100, 5000, 100)], accuracy_lst, '.')
    plt.grid()
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.savefig('q_1d.png')


def main(section):
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    idx = np.random.RandomState(0).choice(70000, 11000)
    train_data = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test_data = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    if section == "b":
        q_b(train_data, train_labels, test_data, test_labels)
    if section == "c":
        q_c(train_data, train_labels, test_data, test_labels)
    if section == "d":
        q_d(train_data, train_labels, test_data, test_labels)


if __name__ == "__main__":
    section = ''
    choices_lst = ['b', 'c', 'd', 'q']
    while section != 'q':
        print "Please choose a section: {b,c,d}, or q to exit"
        section = str(raw_input()).lower()
        if section not in choices_lst:
            print "Bad choice, try again"
        main(section)
    exit(1)
