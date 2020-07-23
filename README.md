# K-MeansClusteringPythonGoPharo
Serial and parallel implementation of k-means clustering vector quantization method in Python and Go, and visualization in Pharo.

## Problem description 
k-Means clustering is an <b>unsupervised</b> machine learning algorithm that finds a fixed number (k) of clusters in a set of data. A <b>cluster</b> is a group of data points that are organized together due to similarities in their input features. When using a K-Means algorithm, a cluster is defined by a <b>centroid</b>, which is a point at the center of a cluster. Every point in a data set is part of the cluster whose centroid is most closely located. So, k-Means finds k number of centroids, and then assigns all data points to the closest cluster, with the aim of keeping the centroids small (we tend to minimize the distance between points in one cluster, so that they make compact ensemble and to maximize distance between different clusters).<br><br>
Given a set of observations (x1, x2, ..., xn), where each observation is a d-dimensional real vector, k-means clustering aims to partition the n observations into k (≤ n) sets S = {S1, S2, ..., Sk} so as to minimize the within-cluster sum of squares. Formally, the objective is to find:
<p align="center">
  <img src="https://github.com/NikolaZubic/K-MeansClusteringPythonGoPharo/blob/master/utils/images/8dc15ec63e0676fc07e790f61efd89484a6b7922.svg">
</p>
where μi is the mean of points in Si. This is equivalent to minimizing the pairwise squared deviations of points in the same cluster:
<p align="center">
  <img src="https://github.com/NikolaZubic/K-MeansClusteringPythonGoPharo/blob/master/utils/images/9fb2388a00fcf4f1df3117883fccd0c4028da33d.svg">
</p>
<br>

## Sequential approach 
1. Cluster the data into k groups where k  is predefined<br>
2. Select k points at random as cluster centers<br>
3. Assign objects to their closest cluster center according to some distance function (for example <i>Euclidean distance</i>)<br>
4. Calculate the centroid or mean of all objects in each cluster<br>
5. Repeat steps 2, 3 and 4 until the same points are assigned to each cluster in consecutive rounds<br>

Finding the optimal solution to the k-means clustering problem for observations in d dimensions is:<br>
* NP-hard in general Euclidean space (of d dimensions) even for two clusters
* NP-hard for a general number of clusters k even in the plane
* if k and d (the dimension) are fixed, the problem can be exactly solved in time O(n^(dk + 1)), where n is the number of entities to be clustered

## Parallel approach 
Main <b>motivation</b> for parallel approach is the fact that k-means clustering performance decreases when we increase number of training examples and when we decide to have a larger k.<br>
<i><b>The main objective</b></i> of this project is to improve performance of k-Means clustering algorithm by splitting training examples into multiple partitions and then we calculate distances and assign clusters in parallel. After that, cluster assignments from each partition are combined to check if clusters changed. For iteration I, if clusters changed in iteration (I - 1), we need to recalculate centroids, else we are done.<br>
The whole process is shown on the following digaram (generated with: [Flowchart Maker](https://app.diagrams.net/)):
<p align="center">
  <img src="https://github.com/NikolaZubic/K-MeansClusteringPythonGoPharo/blob/master/utils/images/parallelApproachDigaram.png">
</p>
Useful reference: https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Chandramohan-Fall-2012-CSE633.pdf

## Programs & libraries needed in order to run this project 
// TO DO

## How to run?
// TO DO

## Results
// TO DO
