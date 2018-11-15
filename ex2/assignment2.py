#################################
# Your name: Ran Tuchman
#################################

import numpy as np
import scipy as sc
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from intervals import find_best_interval
import random as rnd
from sklearn.model_selection import train_test_split
plt.ioff()  # Turn interactive plotting off


class Assignment2(object):
    """Assignment 2 skeleton.

        Please use these function signatures for this assignment and submit this file, together with the intervals.py.
        """

    def sample_from_D(self, m):
        """Sample m data samples from D.
                Input: m - an integer, the size of the data sample.

                Returns: np.ndarray of shape (m,2) :
                        A two dimensional array of size m that contains the pairs where drawn from the distribution P.
                """
        p_distribution = np.zeros(shape=(m, 2))
        x = [rnd.uniform(0, 1) for _ in range(m)]

        for j in range(m):
            p_distribution[j][0] = x[j]
            if (0 <= x[j] <= 0.2) or (0.4 <= x[j] <= 0.6) or (0.8 <= x[j] <= 1):
                p_distribution[j][1] = bernoulli.rvs(0.8)
            else:
                p_distribution[j][1] = bernoulli.rvs(0.1)
        return p_distribution

    def draw_sample_intervals(self, m, k):
        """
                Plots the data as asked in (a) i ii and iii.
                Input: m - an integer, the size of the data sample.
                       k - an integer, the maximum number of intervals.

                Returns: None.
                """
        p = self.sample_from_D(m)
        sorted_p = sorted(p, key=lambda x_y_point: x_y_point[0])
        zero_points = [x[0] for x in p if x[1] == 0]
        one_points = [x[0] for x in p if x[1] == 1]
        plt.plot(one_points, [1 for _ in range(len(one_points))], 'o', label='one')
        plt.plot(zero_points, [0 for _ in range(len(zero_points))], 'o', label='zero')
        plt.axvline(0.2, color='r', linestyle='--', linewidth=1.0)
        plt.axvline(0.4, color='r', linestyle='--', linewidth=1.0)
        plt.axvline(0.6, color='r', linestyle='--', linewidth=1.0)
        plt.axvline(0.8, color='r', linestyle='--', linewidth=1.0)
        plt.axis([0, 1, -0.1, 1.1])

        intervals, empirical_error = find_best_interval([x[0] for x in sorted_p], [y[1] for y in sorted_p], k)

        for i in range(len(intervals)):
            interval_points = np.linspace(intervals[i][0], intervals[i][1], 100)
            plt.plot(interval_points, [-0.09 for _ in range(100)], color='g',
                     linewidth=5.0)
            plt.annotate("{:.2f}".format(interval_points[0]),(interval_points[0], -0.1))
            plt.annotate("{:.2f}".format(interval_points[99]),(interval_points[99], -0.1))

        plt.legend(loc='best')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.title("best intervals (m={},k={})".format(m, k))
        plt.savefig("section a - draw_sample_intervals")
        plt.close()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
                Calculates the empirical error and the true error.
                Plots the average empirical and true errors.
                Input: m_first - an integer, the smallest size of the data sample in the range.
                       m_last - an integer, the largest size of the data sample in the range.
                       step - an integer, the difference between the size of m in each loop.
                       k - an integer, the maximum number of intervals.
                       T - an integer, the number of times the experiment is performed.

                Returns: np.ndarray of shape (n_steps,2).
                    A two dimensional array that contains the average empirical error
                    and the average true error for each m in the range accordingly.
                """
        n_steps = int((m_last - m_first)/step) + 1
        true_error_list = [0 for _ in (range(n_steps))]
        empirical_error_list = [0 for _ in (range(n_steps))]
        m = range(m_first, m_last + step, step)

        for t in range(T):
            for j in range(n_steps):
                p = self.sample_from_D(m[j])
                sorted_p = sorted(p, key=lambda x_y_point: x_y_point[0])
                intervals, empirical_error = find_best_interval([x[0] for x in sorted_p], [y[1] for y in sorted_p], k)
                empirical_error_list[j] += empirical_error/m[j]
                true_error_list[j] += calculate_true_error(intervals)

        errors_array = np.zeros(shape=(n_steps, 2))
        for i in range(n_steps):
            empirical_error_list[i] /= T
            true_error_list[i] /= T
            errors_array[i][0] = empirical_error_list[i]
            errors_array[i][1] = true_error_list[i]

        plt.plot(m, true_error_list, color='r', label='true error')
        plt.plot(m, empirical_error_list, color='b', label='empirical error')
        plt.ylabel('error')
        plt.xlabel('m')
        plt.legend(loc='best')
        plt.title("empirical and true error as function of m")
        plt.savefig("section c - experiment_m_range_erm")
        plt.close()

        return errors_array

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,20.
                Plots the empirical and true errors as a function of k.
                Input: m - an integer, the size of the data sample.
                       k_first - an integer, the maximum number of intervals in the first experiment.
                       m_last - an integer, the maximum number of intervals in the last experiment.
                       step - an integer, the difference between the size of k in each experiment.

                Returns: The best k value (an integer) according to the ERM algorithm.
                """
        n_steps = int((k_last - k_first)/step) + 1
        true_error_list = [0 for _ in (range(n_steps))]
        empirical_error_list = [0 for _ in (range(n_steps))]
        k_list = range(k_first, k_last + step, step)
        p = self.sample_from_D(1500)
        sorted_p = sorted(p, key=lambda x_y_point: x_y_point[0])

        for k in k_list:
            intervals, empirical_error = find_best_interval([x[0] for x in sorted_p], [y[1] for y in sorted_p], k)
            empirical_error_list[k-1] = empirical_error / 1500
            true_error_list[k-1] += calculate_true_error(intervals)

        plt.plot(k_list, true_error_list, color='r', label='true error')
        plt.plot(k_list, empirical_error_list, color='b', label='empirical error')
        plt.ylabel('error')
        plt.xlabel('k')
        plt.legend(loc='best')
        plt.title("empirical and true error as function of k")
        plt.savefig("section d - experiment_k_range_erm")
        plt.close()

        best = 1
        best_k = 0
        for k in range(n_steps):
            if empirical_error_list[k] < best:
                best = empirical_error_list[k]
                best_k = k+1

        return best_k


    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
                Plots additionally the penalty for the best ERM hypothesis.
                and the sum of penalty and empirical error.
                Input: m - an integer, the size of the data sample.
                       k_first - an integer, the maximum number of intervals in the first experiment.
                       m_last - an integer, the maximum number of intervals in the last experiment.
                       step - an integer, the difference between the size of k in each experiment.

                Returns: The best k value (an integer) according to the SRM algorithm.
                """
        n_steps = int((k_last - k_first)/step) + 1
        true_error_list = [0 for _ in (range(n_steps))]
        empirical_error_list = [0 for _ in (range(n_steps))]
        penalty_list = [0 for _ in (range(n_steps))]
        penalty_plus_es_list = [0 for _ in (range(n_steps))]
        k_list = range(k_first, k_last + step, step)
        p = self.sample_from_D(m)
        sorted_p = sorted(p, key=lambda x_y_point: x_y_point[0])

        for k in k_list:
            intervals, empirical_error = find_best_interval([x[0] for x in sorted_p], [y[1] for y in sorted_p], k)
            empirical_error_list[k-1] = empirical_error / m
            true_error_list[k-1] += calculate_true_error(intervals)
            penalty_arg = (8/m)*(2*k*sc.log((2*sc.e*m)/(2*k))+sc.log(4/0.1))  # VCDim = 2k
            penalty_list[k-1] = sc.sqrt(penalty_arg)
            penalty_plus_es_list[k-1] = penalty_list[k-1] + empirical_error_list[k-1]

        plt.plot(k_list, true_error_list, color='r', label='true error')
        plt.plot(k_list, empirical_error_list, color='b', label='empirical error')
        plt.plot(k_list, penalty_list, color='k', label='penalty')
        plt.plot(k_list, penalty_plus_es_list, color='g', label='empirical error + penalty')
        plt.xlabel('k')
        plt.legend(loc='best')
        plt.title("structural risk minimization")
        plt.savefig("section e - experiment_k_range_srm")
        plt.close()

        best = 1
        best_k = 0
        for k in range(n_steps):
            if penalty_plus_es_list[k] < best:
                best = penalty_plus_es_list[k]
                best_k = k+1

        return best_k


    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
                Chooses the best hypothesis based on 3 experiments.
                Input: m - an integer, the size of the data sample.
                       T - an integer, the number of times the experiment is performed.

                Returns: The best k value (an integer) found by the cross validation algorithm.
                """
        p = self.sample_from_D(m)
        holdout_error_list = np.zeros(10)

        for _ in range(T):
            train_points, test_points = train_test_split(p, test_size=0.2, random_state=42)
            train_points = sorted(train_points, key=lambda x: x[0])
            for k in range(1, 11):
                intervals, empirical_error = find_best_interval([x[0] for x in train_points], [y[1] for y in train_points], k)
                holdout_error_list[k-1] += (calc_holdout_error(test_points, intervals) / T)

        best = 1
        best_k = 0
        for k in range(10):
            if holdout_error_list[k] < best:
                best = holdout_error_list[k]
                best_k = k+1

        return best_k


def calc_intervals_intersection(interval1, interval2):
    if interval1[0] > interval2[0]:
        interval1, interval2 = interval2, interval1
    if interval1[1] <= interval2[0]:
        return 0
    return min(interval1[1], interval2[1]) - interval2[0]


def calc_intervals_complement(intervals):
    intervals_copy = intervals[:]
    intervals_copy.insert(0, (0, 0))
    intervals_copy.append((1, 1))
    complements = []
    for i in range(len(intervals_copy) - 1):
        complements.append((intervals_copy[i][1], intervals_copy[i + 1][0]))
    return complements


def calculate_true_error(h):
    h_complement = calc_intervals_complement(h)
    my_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
    my_intervals_complement = calc_intervals_complement([(0, 0.2), (0.4, 0.6), (0.8, 1)])
    intersections_list = [0, 0, 0, 0]
    for interval in h_complement:
        for other_interval in my_intervals:
            intersections_list[0] += calc_intervals_intersection(interval, other_interval)
    for interval in h_complement:
        for other_interval in my_intervals_complement:
            intersections_list[1] += calc_intervals_intersection(interval, other_interval)
    for interval in h:
        for other_interval in my_intervals:
            intersections_list[2] += calc_intervals_intersection(interval, other_interval)
    for interval in h:
        for other_interval in my_intervals_complement:
            intersections_list[3] += calc_intervals_intersection(interval, other_interval)
    error = 0.8*intersections_list[0] + 0.1*intersections_list[1] + \
            0.2*intersections_list[2] + 0.9*intersections_list[3]
    return error


def calc_holdout_error(samples, intervals):
    intervals_complement = calc_intervals_complement(intervals)
    zero_points = [x[0] for x in samples if x[1] == 0]
    one_points = [x[0] for x in samples if x[1] == 1]
    error = 0.0
    for point in one_points:
        for inter in intervals_complement:
            if inter[0] <= point <= inter[1]:
                error += 1
                break
    for point in zero_points:
        for inter in intervals:
            if inter[0] <= point <= inter[1]:
                error += 1
                break

    return error/len(samples)


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
