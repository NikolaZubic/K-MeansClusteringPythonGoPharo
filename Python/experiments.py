from kMeansClustering import KMeansClustering
import pandas as pd


def sequential_clustering_experiment():
    data_csv_path = "testData/College.csv"
    sequential_results_path = ""
    parallel_results_path = ""
    separator = ","

    data_frame = pd.read_csv(data_csv_path, sep=separator)
    data_frame = data_frame.drop(['UniName', 'Private'], axis=1)

    for i in range(2, 15):
        k_means_clustering = KMeansClustering(csv_df=data_frame,
                                              sequential_generate_results_path=sequential_results_path,
                                              parallel_generate_results_path=parallel_results_path, k=i, max_iter=1000)
        KMeansClustering.initialize_centroids(k_means_clustering)
        print("CENTROIDS: " + str(k_means_clustering.centroids_indexes))
        _ = k_means_clustering.run_sequential_clustering()
        print(61 * "=")


def weak_scaling():
    data_csv_path = "testData/College.csv"
    sequential_results_path = ""
    parallel_results_path = ""
    separator = ","
    number_of_tasks = 14

    data_frame = pd.read_csv(data_csv_path, sep=separator)
    data_frame = data_frame.drop(['UniName', 'Private'], axis=1)

    k_means_clustering = KMeansClustering(csv_df=data_frame, sequential_generate_results_path=sequential_results_path,
                                          parallel_generate_results_path=parallel_results_path, k=14, max_iter=100)
    #KMeansClustering.initialize_centroids(k_means_clustering)
    k_means_clustering.centroids_indexes = [3, 5, 86, 109, 111, 243, 299, 375, 390, 475, 494, 538, 598, 670]

    data_points_parallel = k_means_clustering.run_parallel_clustering(number_of_tasks=int(number_of_tasks))


def strong_scaling():
    data_csv_path = "testData/College.csv"
    sequential_results_path = "results/secondExample/sequential"
    parallel_results_path = "results/secondExample/parallel"
    separator = ","

    data_frame = pd.read_csv(data_csv_path, sep=separator)
    data_frame = data_frame.drop(['UniName', 'Private'], axis=1)

    for i in range(2, 15):
        k_means_clustering = KMeansClustering(csv_df=data_frame,
                                              sequential_generate_results_path=sequential_results_path,
                                              parallel_generate_results_path=parallel_results_path, k=14, max_iter=100)
        #KMeansClustering.initialize_centroids(k_means_clustering)
        k_means_clustering.centroids_indexes = [3, 5, 86, 109, 111, 243, 299, 375, 390, 475, 494, 538, 598, 670]

        data_points_parallel = k_means_clustering.run_parallel_clustering(number_of_tasks=int(i))


if __name__ == '__main__':
    print("Starting sequential clustering experiment...")
    #sequential_clustering_experiment()
    print("Sequential clustering experiment finished!")

    print("Starting weak parallel clustering experiment...")
    #weak_scaling()
    print("Weak parallel clustering experiment finished!")

    print("Starting strong parallel clustering experiment...")
    #strong_scaling()
    print("Strong parallel clustering experiment finished!")
