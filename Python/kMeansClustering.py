"""
Sequential and parallel implementation of k-means clustering algorithm.

Author: Nikola Zubic
"""

from helperMethods import time_measure, centroids_initialization, euclidean_distance
import numpy as np
import pandas as pd
import sys
import multiprocessing
import itertools
import scipy.spatial.distance as distance


class KMeansClustering(object):
    """
    csv_path: csv_path to read csv as Pandas DataFrame object
    sequential_generate_results_path: path where sequential results for visualization with Pharo will be saved
    parallel_generate_results_path: path where parallel results for visualization with Pharo will be saved
    k: fixed number of clusters
    max_iter: maximum number of iterations, default: 100
    centroids_indexes: indexes of randomly chosen centroids
    """
    def __init__(self, csv_df, sequential_generate_results_path, parallel_generate_results_path, k, max_iter=100,
                 centroids_indexes=None):
        self.csv_df = csv_df
        self.sequential_generate_results_path = sequential_generate_results_path
        self.parallel_generate_results_path = parallel_generate_results_path
        self.k = k
        self.max_iter = max_iter
        self.centroids_indexes = centroids_indexes

    @staticmethod
    def initialize_centroids(self):
        self.centroids_indexes = centroids_initialization(k=self.k, m=range(self.csv_df.shape[0]))

    @time_measure
    def run_sequential_clustering(self):
        """
        A cluster is a group of data points that are organized together due to similarities in their input features.
        When using a K-Means algorithm, a cluster is defined by a centroid, which is a point at the center of a cluster.
        Every point in a data set is part of the cluster whose centroid is most closely located.
        :return: data points representing which training example belongs to which cluster
        """
        csv_df_copy = self.csv_df

        data_set_size = self.csv_df.shape[0]

        data_points = np.zeros(data_set_size, dtype=int)

        # Access a group of rows and columns by label(s) or a boolean array
        centroids = self.csv_df.loc[self.centroids_indexes]  # from all data, centroids are chosen at given indexes

        for iter in range(0, self.max_iter):
            old_data_points = data_points.copy()

            for i in range(data_set_size):
                distances = []
                distances = [euclidean_distance(centroids.loc[self.centroids_indexes[j]], self.csv_df.loc[i])
                             for j in range(self.k)]
                data_points[i] = distances.index(min(distances))

            # If there is no change, we are done
            check_if_equal = np.array(data_points) == np.array(old_data_points)
            equal = check_if_equal.all()

            if equal:
                break

            # If there was change, we need to recalculate centroids
            for c in range(self.k):
                centroids.loc[self.centroids_indexes[c]] = self.csv_df.loc[data_points == c].mean(axis=0)
                csv_df_copy.loc[self.centroids_indexes[c]] = self.csv_df.loc[data_points == c].mean(axis=0)

            """
            Output format: x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...) | ... |
                           x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...)
            """

            if sequential_results_path != "":
                f = open(self.sequential_generate_results_path + "/current_state_" + str(iter) + ".txt", "a")

                counter = 0
                for coordinates, cluster, dp_size in zip(csv_df_copy.values, data_points, range(data_points.size)):
                    s = 0

                    if counter in self.centroids_indexes:
                        s = 1

                    if dp_size == data_points.size - 1:
                        f.write("{} {} {} {}".format(coordinates[0], coordinates[1], s, cluster))
                    else:
                        f.write("{} {} {} {}|".format(coordinates[0], coordinates[1], s, cluster))

                    counter += 1
                    s = 0

        return data_points

    @staticmethod
    def find_cluster(data_points, centroids, metric='euclidean'):
        distances_between_points = distance.cdist(XA=data_points, XB=centroids, metric=metric)
        return np.argmin(distances_between_points, axis=1)

    def parallelize(self, centroids, splitted_data, number_of_tasks):
        distributed_centroids_copy = itertools.repeat(centroids)

        arguments_for_find_cluster = zip(splitted_data, distributed_centroids_copy)

        pool = multiprocessing.Pool(processes=number_of_tasks)

        split_data_points = pool.starmap(KMeansClustering.find_cluster, arguments_for_find_cluster)

        pool.close()
        pool.join()

        data_points = np.concatenate(split_data_points)

        return data_points

    @time_measure
    def run_parallel_clustering(self, number_of_tasks):
        """
        Main motivation for parallel approach is the fact that k-means clustering performance decreases when we increase
        number of training examples and when we decide to have a larger k.
        The main objective of this project is to improve performance of k-Means clustering algorithm by splitting
        training examples into multiple partitions and then we calculate distances and assign clusters in parallel.
        After that, cluster assignments from each partition are combined to check if clusters changed. For iteration I,
        if clusters changed in iteration (I - 1), we need to recalculate centroids, else we are done.
        :return: data points representing which training example belongs to which cluster
        """
        csv_df_copy = self.csv_df

        data_set_size = self.csv_df.shape[0]

        data_points = np.zeros(data_set_size, dtype=int)

        # Access a group of rows and columns by label(s) or a boolean array
        centroids = self.csv_df.loc[self.centroids_indexes]  # from all data, centroids are chosen at given indexes

        # Split an array into multiple sub-arrays.
        split_data = np.array_split(self.csv_df, number_of_tasks)

        for iter in range(0, self.max_iter):
            old_data_points = data_points.copy()

            data_points = self.parallelize(centroids=centroids, splitted_data=split_data,
                                           number_of_tasks=number_of_tasks)

            # If there is no change, we are done
            check_if_equal = np.array(data_points) == np.array(old_data_points)
            equal = check_if_equal.all()

            if equal:
                break

            # If there was change, we need to recalculate centroids
            for c in range(self.k):
                centroids.loc[self.centroids_indexes[c]] = self.csv_df.loc[data_points == c].mean(axis=0)
                csv_df_copy.loc[self.centroids_indexes[c]] = self.csv_df.loc[data_points == c].mean(axis=0)

            """
            Output format: x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...) | ... |
            x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...)
            """

            if self.parallel_generate_results_path != "":
                f = open(self.parallel_generate_results_path + "/current_state_" + str(iter) + ".txt", "a")

                counter = 0
                for coordinates, cluster, dp_size in zip(csv_df_copy.values, data_points, range(data_points.size)):
                    s = 0

                    if counter in self.centroids_indexes:
                        s = 1

                    if dp_size == data_points.size - 1:
                        f.write("{} {} {} {}".format(coordinates[0], coordinates[1], s, cluster))
                    else:
                        f.write("{} {} {} {}|".format(coordinates[0], coordinates[1], s, cluster))

                    counter += 1
                    s = 0

        return data_points


if __name__ == '__main__':
    data_csv_path = sys.argv[1]
    sequential_results_path = sys.argv[2]
    parallel_results_path = sys.argv[3]
    separator = sys.argv[4]
    number_of_tasks = sys.argv[5]

    data_frame = pd.read_csv(data_csv_path, sep=separator)
    data_frame = data_frame.drop(['UniName', 'Private'], axis=1)

    """
    data_frame = data_frame[['Apps', 'Accept']]
    python kMeansClustering.py testData/College.csv results/secondExample/sequential results/secondExample/parallel , 4
    """

    """
    python kMeansClustering.py testData/College.csv results/thirdExample/sequential results/thirdExample/parallel , 5
    """

    k_means_clustering = KMeansClustering(csv_df=data_frame, sequential_generate_results_path=sequential_results_path,
                                          parallel_generate_results_path=parallel_results_path, k=4, max_iter=100)
    KMeansClustering.initialize_centroids(k_means_clustering)

    data_points_sequential = k_means_clustering.run_sequential_clustering()
    data_points_parallel = k_means_clustering.run_parallel_clustering(number_of_tasks=int(number_of_tasks))

    check_if_equal = np.array(data_points_sequential) == np.array(data_points_parallel)
    equal = check_if_equal.all()

    if not equal:
        print("[ERROR]: Algorithms aren't equal!")
