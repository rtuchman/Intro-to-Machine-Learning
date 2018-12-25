#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
plt.ioff()  # Turn interactive plotting off
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """
    n = len(X_train)
    alphas = []
    h_s = []

    # Initialize distribution.
    D = np.array([1.0 / n for _ in range(n)])
    for t in range(T):
        print("T ={}".format(t))
        h_pos, err_pos = WL(X_train, y_train, D, 1)
        h_neg, err = WL(X_train, y_train, D, -1)
        if err_pos < err:
            h_neg = h_pos
            err = err_pos
        alpha = np.log((1 - err) / err) * 0.5

        h_s.append(h_neg)
        alphas.append(alpha)

        # Update distribution.
        distribution = np.array([np.exp(np.negative(alpha * (np.sign(h_neg[2] - X_train[i][h_neg[1]]) * h_neg[0]) * y_train[i])) for i in range(n)])
        z_t = np.dot(D, distribution)
        new_distribution = distribution * (1.0 / z_t)

        D = np.multiply(D, new_distribution)

    return h_s, alphas

def WL(train_data, train_labels, D, prediction):
    n = train_data.shape[0]
    dimentions = train_data.shape[1]
    labeled_data = list(zip(train_data, train_labels, D))

    min_err = float('inf')
    theta = train_data[0][0] - 1
    dim_idx = 0

    for k in range(dimentions):
        dim_sort = sorted(labeled_data, key=lambda x: x[0][k])
        add_one_label = [(dim_sort[-1][0][k] + 1) for _ in range(dimentions)]
        dim_sort.append((add_one_label, float('inf'), float('inf')))
        curr_error = np.sum([p[2] for p in dim_sort if p[1] == prediction])

        if curr_error < min_err:
            min_err = curr_error
            theta = dim_sort[0][0][k] - 1
            dim_idx = k

        # Iterate through sorted points.
        for j in range(n):
            curr_error = curr_error - (prediction * (dim_sort[j][1] * dim_sort[j][2]))

            if (curr_error < min_err) and (dim_sort[j][0][k] != dim_sort[j + 1][0][k]):
                min_err = curr_error
                theta = 0.5 * (dim_sort[j][0][k] + dim_sort[j + 1][0][k])
                dim_idx = k

    return (prediction, dim_idx, theta), min_err

def calc_error(data, labels, hypothesis):
    error = 0
    for i in range(len(data)):
        error += 0 if hypothesis(data[i]) == labels[i] else 1
    return float(error) / len(data)

def calc_loss(X, y, h, alphas, t):
    loss = 0
    for j in range(len(X)):
        exponent = 0
        for i in range(t+1):
            exponent += alphas[i] * calc_h_of_x(h[i], X[j])
        exponent *= -y[j]
        loss += np.exp(exponent)

    return loss / len(X)

def calc_h_of_x(h, x):
    if x[h[1]] < h[2]:
        return h[0]
    return -h[0]

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T=10
    hypotheses, alphas = run_adaboost(X_train, y_train, T)
    train_errors = []
    test_errors = []
    for t in range(T):
        curr_boosted_h = lambda x: np.sign(
            np.sum([(np.sign(hypotheses[i][2] - x[hypotheses[i][1]]) * alphas[i]) for i in range(t+1)]))
        train_errors.append(calc_error(X_train, y_train, curr_boosted_h))
        test_errors.append(calc_error(X_test, y_test, curr_boosted_h))

    t_list = list(range(80))
    plt.plot(t_list, train_errors, label='train_error')
    plt.plot(t_list, test_errors, label='test_error')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('error')
    plt.savefig('q_1a.png')

    train_loss = []
    test_loss = []
    for t in range(T):
        train_loss.append(calc_loss(X_train, y_train, hypotheses, alphas, t))
        test_loss.append(calc_loss(X_test, y_test, hypotheses, alphas, t))

    t_list = list(range(80))
    plt.plot(t_list, train_loss, label='train_loss')
    plt.plot(t_list, test_loss, label='test_loss')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('loss')
    plt.savefig('q_1c.png')












if __name__ == '__main__':
    main()



