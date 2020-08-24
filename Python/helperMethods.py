from time import time, sleep
import numpy as np
import random


def time_measure(method):
    """
    Decorator that allows us to measure execution time of a certain method/function in seconds.
    In this case, we will measure time length of sequential and parallel approach in Python.
    :param method: method/function for which we will measure execution time
    :return: execution_time method which returns given result as formatted string
    """
    def execution_time(*args, **kwargs):
        start_time = time()
        method_execution = method(*args, **kwargs)
        end_time = time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = round(end_time - start_time, 5)
        else:
            print(str(round(end_time - start_time, 5)) + " seconds")

        return method_execution

    return execution_time


@time_measure
def test_time_measure():
    print("Started execution...")
    sleep(5.3253)
    print("Finished with method execution.")


def euclidean_distance(x, y):
    """
    The Euclidean distance measure is generalized by Minkowski distance metric:
    d(x, y) = (sum_from_k=1_to_n(abs(x_k - y_k))**r) ** (1 / r),
    where r is a parameter. For r = 2, we get L2 norm (Euclidean distance)
    :param x: dot (centroid)
    :param y: dot (data-point)
    :return: l2 norm between points x and y
    """
    return np.linalg.norm(x - y)


def random_sampling_without_replacement(population, k):
    """
    Used for random sampling without replacement.
    :param population: generally a list, in this case a given range of number of training examples
    :param k: fixed number (k) of clusters in a set of data
    :return: k length list of unique elements chosen from the population sequence or set
    """
    return random.sample(population, k)


def centroids_initialization(k, m):
    """
    Function that will be used for centroids initialization, based on given data-set.
    :param k: fixed number (k) of clusters in a set of data
    :param m: number of training examples
    :return: indexes for centroids
    """
    centroid_indexes = random_sampling_without_replacement(population=m, k=k)
    centroid_indexes.sort()
    return centroid_indexes


if __name__ == '__main__':
    test_time_measure()
    print(centroids_initialization(3, range(100)))
    a = np.zeros(440, dtype=int)
    print(a.shape)
