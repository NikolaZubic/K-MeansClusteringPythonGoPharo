package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/stat"

	"gonum.org/v1/gonum/mat"

	"github.com/kniren/gota/dataframe"
)

/*
KMeansClustering type is equivalent to KMeansClustering class in Python implementation

csvDf: csv_path to read csv as Pandas DataFrame object
sequentialGenerateResultsPath: path where sequential results for visualization with Pharo will be saved
parallelGenerateResultsPath: path where parallel results for visualization with Pharo will be saved
k: fixed number of clusters
maxIter: maximum number of iterations, default: 100
centroidsIndexes: indexes of randomly chosen centroids
*/
type KMeansClustering struct {
	csvDf                         dataframe.DataFrame
	sequentialGenerateResultsPath string
	parallelGenerateResultsPath   string
	k                             int
	maxIter                       int
	centroidsIndexes              []int16
}

// Round float64 number and return int number
func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

// Round float64 number to a given int precision
func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}

// Function used for measuring sequential and parallel's implementations time of execution
func timeMeasure() func() {
	startTime := time.Now()

	return func() {
		fmt.Print(toFixed(time.Since(startTime).Seconds(), 5))
		fmt.Println(" seconds")
	}
}

// euclideanDistance computes L2 norm between two vectors
func euclideanDistance(firstVector, secondVector []float64) (float64, error) {
	distance := 0.

	for ii := range firstVector {
		distance += (firstVector[ii] - secondVector[ii]) * (firstVector[ii] - secondVector[ii])
	}

	return math.Sqrt(distance), nil
}

// Remove element at given index s from slice
func remove(slice []int16, s int) []int16 {
	return append(slice[:s], slice[s+1:]...)
}

/*
Given a population (generally a list, in this case a given range of number of training examples) and fixed number (k)
of clusters in a set of data, returns k-length list of unique elements chosen from the population sequence/set
*/
func randomSamplingWithoutReplacement(population []int16, k int) []int16 {
	subset := []int16{}

	for true {
		randomIndex := rand.Intn(len(population))
		pick := population[randomIndex]
		subset = append(subset, pick)
		population = remove(population, randomIndex)

		if len(subset) == k {
			break
		}
	}

	return subset
}

// From a given dataset where we have m number of training examples (0 to length-1), choose k length list of them
func centroidsInitialization(k int, m []int16) []int16 {
	population := randomSamplingWithoutReplacement(m, k)

	sort.SliceStable(population, func(i, j int) bool {
		return population[i] < population[j]
	})

	return population
}

// For a given number N, create a list [0, 1, ... , N-1]
func rangeList(N int16) []int16 {
	retVal := []int16{}
	for i := int16(0); i < N; i++ {
		retVal = append(retVal, i)
	}

	return retVal
}

// Initialize centroids indexes collection, to prevent it from being nil
func (kmeans *KMeansClustering) initializeCentroidsIndexesCollection() {
	kmeans.centroidsIndexes = []int16{}
}

// Function that initializesCentroids and sets centroidsIndexes attribute to a given array of indexes
func (kmeans *KMeansClustering) initializeCentroids() {
	kmeans.initializeCentroidsIndexesCollection()

	populationRange := rangeList(int16(kmeans.csvDf.Nrow()))
	population := centroidsInitialization(kmeans.k, populationRange)

	for _, element := range population {
		kmeans.centroidsIndexes = append(kmeans.centroidsIndexes, element)
	}

}

// Given a csv dataFrame represented with rows and columns, transform it into GoNum matrix, so we can operate on it
func makeMatrix(df *dataframe.DataFrame) *mat.Dense {
	retVal := mat.NewDense(df.Nrow(), df.Ncol(), nil)

	for i := 0; i < df.Nrow(); i++ {
		for j := 0; j < df.Ncol(); j++ {
			retVal.Set(i, j, df.Elem(i, j).Float())
		}
	}

	return retVal
}

// Function that transforms given mat.Dense to 2D-array of float64 numbers so that we can calculate Euclidean distance
func centroidsListOfArrays(centroidsMatrix *mat.Dense) [][]float64 {
	centroidsSize, features := centroidsMatrix.Dims()

	retVal := [][]float64{}

	for i := 0; i < centroidsSize; i++ {
		concreteCentroid := []float64{}

		for j := 0; j < features; j++ {
			concreteCentroid = append(concreteCentroid, centroidsMatrix.At(i, j))
		}

		retVal = append(retVal, concreteCentroid)
	}

	return retVal
}

/*
For a given data-set represented as matrix and number k, returns an array representing number of data-points that belong
to k-th cluster.
For example, if we have 80 datapoints and k=3 clusters and we get an array: [30, 40, 10] this means that 30 datapoints
belong to first cluster, 40 to second cluster and 10 to the third cluster.
*/
func (kmeans *KMeansClustering) countDataPointsByClusters(dataPoints *mat.Dense, k int) []int {
	rows, _ := dataPoints.Dims()

	retVal := []int{}

	for i := 0; i < k; i++ {
		retVal = append(retVal, 0)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < k; j++ {
			if dataPoints.At(i, 0) == float64(j) {
				val := retVal[j]
				val++
				retVal[j] = val
			}
		}
	}

	return retVal
}

// Groups data in N matrices where i-th matrix belong to (k + 1)-th cluster
func (kmeans *KMeansClustering) groupDataByCurrentClusters(csvDf *mat.Dense, dataPoints *mat.Dense,
	k int) []*mat.Dense {
	_, cols := csvDf.Dims()

	dataPointsNumbers := kmeans.countDataPointsByClusters(dataPoints, k)

	retVal := make([]*mat.Dense, k)

	counters := make([]int, k)

	for i := 0; i < len(dataPointsNumbers); i++ {
		retVal[i] = mat.NewDense(dataPointsNumbers[i], cols, nil)
		counters[i] = 0
	}

	rows, _ := dataPoints.Dims()

	for i := 0; i < rows; i++ {
		currInd := int(dataPoints.At(i, 0))

		// retVal[currInd] gives corresponding matrix initially filled with zeros
		for j := 0; j < cols; j++ {
			//csvDfLoc = append(csvDfLoc, csvDfMatrix.At(i, j))
			retVal[currInd].Set(counters[currInd], j, csvDf.At(i, j))
		}

		val := counters[currInd]
		val++
		counters[currInd] = val
	}

	return retVal
}

/*Function that writes a given result to file.
Output format: x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...) | ... |
               x_coordinate y_coordinate is_centroid(0-no, 1-yes) centroid_index(0, 1 , 2,...)
*/
func (kmeans *KMeansClustering) writeToFile(filePath string, csvDfCopyMatrix *mat.Dense, dataPoints *mat.Dense,
	indexes []int) {
	rows, _ := csvDfCopyMatrix.Dims()

	file, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {
		fmt.Println(err)
	}

	defer file.Close()

	for i := 0; i < rows; i++ {
		s := 0

		fString := fmt.Sprintf("%f", csvDfCopyMatrix.At(i, 0))
		fString = fString + " " + fmt.Sprintf("%f", csvDfCopyMatrix.At(i, 1))

		for c := 0; c < len(indexes); c++ {
			if indexes[c] == i {
				fString = fString + " " + fmt.Sprintf("%d", 1)
				s = 1
			}
		}

		if s != 1 {
			fString = fString + " " + fmt.Sprintf("%d", 0)
		}

		fString = fString + " " + fmt.Sprintf("%d", int(dataPoints.At(i, 0)))

		if i != (rows - 1) {
			fString = fString + "|"
		}

		_, err2 := file.WriteString(fString)

		if err2 != nil {
			fmt.Println(err2)
		}

		s = 0
		fString = ""
	}

}

/*
A cluster is a group of data points that are organized together due to similarities in their input features.
When using a K-Means algorithm, a cluster is defined by a centroid, which is a point at the center of a cluster.
Every point in a data set is part of the cluster whose centroid is most closely located.
*/
func (kmeans *KMeansClustering) runSequentialClustering() *mat.Dense {
	defer timeMeasure()()

	csvDfMatrix := makeMatrix(&kmeans.csvDf)
	_, csvDfCols := csvDfMatrix.Dims()

	csvDfCopyMatrix := makeMatrix(&kmeans.csvDf)

	dataSetSize := kmeans.csvDf.Nrow()

	dataPoints := mat.NewDense(dataSetSize, 1, nil)

	indexes := []int{}

	for _, element := range kmeans.centroidsIndexes {
		indexes = append(indexes, int(element))
	}

	centroids := kmeans.csvDf.Subset(indexes)
	centroidsMatrix := makeMatrix(&centroids)

	for iter := 0; iter < kmeans.maxIter; iter++ {
		oldDataPoints := mat.DenseCopyOf(dataPoints)

		for i := 0; i < dataSetSize; i++ {
			distances := []float64{}

			csvDfLoc := []float64{}

			for j := 0; j < csvDfCols; j++ {
				csvDfLoc = append(csvDfLoc, csvDfMatrix.At(i, j))
			}

			centroidsList := centroidsListOfArrays(centroidsMatrix)

			for u := 0; u < kmeans.k; u++ {
				euclideanDist, err := euclideanDistance(csvDfLoc, centroidsList[u])

				if err != nil {
					fmt.Println("Error while computing Euclidean distance.")
					return nil
				}

				distances = append(distances, euclideanDist)
			}

			min := distances[0]
			ind := 0

			for in, v := range distances {
				if v < min {
					min = v
					ind = in
				}
			}

			dataPoints.Set(i, 0, float64(ind))

		}

		checkIfEqual := mat.Equal(dataPoints, oldDataPoints)

		if checkIfEqual {
			break
		}

		groupedDataByClusters := kmeans.groupDataByCurrentClusters(csvDfMatrix, dataPoints, kmeans.k)

		// If there was change, we need to recalculate centroids
		for c := 0; c < kmeans.k; c++ {
			currentMatrix := groupedDataByClusters[c]
			currentMatrixRows, currentMatrixCols := currentMatrix.Dims()

			// indexes[c] will give index of current cluster and we will use that to update csvDfCopy
			for col := 0; col < currentMatrixCols; col++ {
				colValues := []float64{}
				for row := 0; row < currentMatrixRows; row++ {
					colValues = append(colValues, currentMatrix.At(row, col))
				}

				colMean := stat.Mean(colValues, nil)

				centroidsMatrix.Set(c, col, colMean)
				csvDfCopyMatrix.Set(indexes[c], col, colMean)
			}
		}

		if kmeans.sequentialGenerateResultsPath != "" {
			filePath := kmeans.sequentialGenerateResultsPath + "/current_state_" + strconv.Itoa(iter) + ".txt"

			kmeans.writeToFile(filePath, csvDfCopyMatrix, dataPoints, indexes)
		}
	}

	return dataPoints

}

func arraySplit(csvDf *mat.Dense, n int) []*mat.Dense {
	/*
		The only difference between these functions is that array_split allows indices_or_sections to be an integer that
		does not equally divide the axis. For an array of length l that should be split into n sections, it returns l % n
		sub-arrays of size l//n + 1 and the rest of size l//n.
	*/
	l, cols := csvDf.Dims()

	retVal := make([]*mat.Dense, n)

	lModN := l % n

	for i := 0; i < n; i++ {
		if i < lModN {
			retVal[i] = mat.NewDense((l/n + 1), cols, nil)
		} else {
			retVal[i] = mat.NewDense((l / n), cols, nil)
		}
	}

	counter := 0

	for i := 0; i < n; i++ {
		currentMatrixRows, currentMatrixCols := retVal[i].Dims()
		for row := 0; row < currentMatrixRows; row++ {
			for col := 0; col < currentMatrixCols; col++ {
				retVal[i].Set(row, col, csvDf.At(counter, col))
			}
			counter++
		}
	}

	return retVal
}

// distance calculation and cluster assignment for one task
func (kmeans *KMeansClustering) findCluster(csvDfMatrix *mat.Dense, centroidsMatrix *mat.Dense,
	syncWaitGroup *sync.WaitGroup, splitDataPoints []*mat.Dense, currentUnit int, dataSetSize int, csvDfCols int) {

	centroidsMatrixCopy := mat.DenseCopyOf(centroidsMatrix)

	for i := 0; i < dataSetSize; i++ {
		distances := []float64{}

		csvDfLoc := []float64{}

		for j := 0; j < csvDfCols; j++ {
			csvDfLoc = append(csvDfLoc, csvDfMatrix.At(i, j))
		}

		centroidsList := centroidsListOfArrays(centroidsMatrixCopy)

		for u := 0; u < kmeans.k; u++ {
			euclideanDist, err := euclideanDistance(csvDfLoc, centroidsList[u])

			if err != nil {
				fmt.Println("Error while computing Euclidean distance.")
				return
			}

			distances = append(distances, euclideanDist)
		}

		min := distances[0]
		ind := 0

		for in, v := range distances {
			if v < min {
				min = v
				ind = in
			}
		}

		splitDataPoints[currentUnit].Set(i, 0, float64(ind))

	}

	// Done decrements the WaitGroup counter by one.
	syncWaitGroup.Done()

}

// Concatenate splitted datapoints to one array / results combination for separate processes
func (kmeans *KMeansClustering) concatenate(splitDataPoints []*mat.Dense) *mat.Dense {
	numberOfRows := 0
	numberOfColumns := 0
	for i := 0; i < len(splitDataPoints); i++ {
		rows, cols := splitDataPoints[i].Dims()
		numberOfRows += rows
		numberOfColumns = cols
	}

	retVal := mat.NewDense(numberOfRows, numberOfColumns, nil)
	counter := 0

	for i := 0; i < len(splitDataPoints); i++ {
		currRows, currCols := splitDataPoints[i].Dims()

		for r := 0; r < currRows; r++ {
			for c := 0; c < currCols; c++ {
				retVal.Set(counter, c, splitDataPoints[i].At(r, c))
			}
			counter++
		}
	}

	return retVal

}

/*
Main motivation for parallel approach is the fact that k-means clustering performance decreases when we increase
number of training examples and when we decide to have a larger k.
The main objective of this project is to improve performance of k-Means clustering algorithm by splitting
training examples into multiple partitions and then we calculate distances and assign clusters in parallel.
After that, cluster assignments from each partition are combined to check if clusters changed. For iteration I,
if clusters changed in iteration (I - 1), we need to recalculate centroids, else we are done.
*/
func (kmeans *KMeansClustering) runParallelClustering(numberOfTasks int) *mat.Dense {
	defer timeMeasure()()

	csvDfMatrix := makeMatrix(&kmeans.csvDf)
	//_, csvDfCols := csvDfMatrix.Dims()

	csvDfCopyMatrix := makeMatrix(&kmeans.csvDf)

	dataSetSize := kmeans.csvDf.Nrow()

	dataPoints := mat.NewDense(dataSetSize, 1, nil)

	indexes := []int{}

	for _, element := range kmeans.centroidsIndexes {
		indexes = append(indexes, int(element))
	}

	centroids := kmeans.csvDf.Subset(indexes)
	centroidsMatrix := makeMatrix(&centroids)

	splitData := arraySplit(csvDfMatrix, numberOfTasks)
	splitDataPoints := make([]*mat.Dense, numberOfTasks)

	for i := 0; i < len(splitDataPoints); i++ {
		currRows, _ := splitData[i].Dims()
		splitDataPoints[i] = mat.NewDense(currRows, 1, nil)
	}

	for iter := 0; iter < kmeans.maxIter; iter++ {
		oldDataPoints := mat.DenseCopyOf(dataPoints)

		/*
			A WaitGroup waits for a collection of goroutines to finish. The main goroutine calls Add to set the number of
			goroutines to wait for. Then each of the goroutines runs and calls Done when finished. At the same time, Wait
			can be used to block until all goroutines have finished.
		*/
		var syncWaitGroup sync.WaitGroup

		for i := 0; i < numberOfTasks; i++ {
			dsSize, csvCols := splitData[i].Dims()
			syncWaitGroup.Add(1)
			go kmeans.findCluster(splitData[i], centroidsMatrix, &syncWaitGroup, splitDataPoints, i, dsSize, csvCols)
		}

		// Wait blocks until the WaitGroup counter is zero
		syncWaitGroup.Wait()

		dataPoints = kmeans.concatenate(splitDataPoints)

		checkIfEqual := mat.Equal(dataPoints, oldDataPoints)

		if checkIfEqual {
			break
		}

		groupedDataByClusters := kmeans.groupDataByCurrentClusters(csvDfMatrix, dataPoints, kmeans.k)

		// If there was change, we need to recalculate centroids
		for c := 0; c < kmeans.k; c++ {
			currentMatrix := groupedDataByClusters[c]
			currentMatrixRows, currentMatrixCols := currentMatrix.Dims()

			// indexes[c] will give index of current cluster and we will use that to update csvDfCopy
			for col := 0; col < currentMatrixCols; col++ {
				colValues := []float64{}
				for row := 0; row < currentMatrixRows; row++ {
					colValues = append(colValues, currentMatrix.At(row, col))
				}

				colMean := stat.Mean(colValues, nil)

				centroidsMatrix.Set(c, col, colMean)
				csvDfCopyMatrix.Set(indexes[c], col, colMean)
			}
		}

		if kmeans.parallelGenerateResultsPath != "" {

			filePath := kmeans.parallelGenerateResultsPath + "/current_state_" + strconv.Itoa(iter) + ".txt"

			kmeans.writeToFile(filePath, csvDfCopyMatrix, dataPoints, indexes)
		}

	}

	return dataPoints

}

func sequentialClusteringExperiment() {
	b, err := ioutil.ReadFile(
		"C:/Users/Nikola Zubic/Desktop/NTP projekat/K-MeansClusteringPythonGoPharo/Python/testData/College.csv")
	if err != nil {
		fmt.Print(err)
	}

	csvStr := string(b) // convert content to a 'string'

	df := dataframe.ReadCSV(strings.NewReader(csvStr)) // pass an io.Reader to the ReadCSV

	df = df.Select([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})

	kMeans := KMeansClustering{
		csvDf:                         df,
		sequentialGenerateResultsPath: "",
		parallelGenerateResultsPath:   "",
		k:                             14,
		maxIter:                       1000,
		centroidsIndexes:              nil,
	}

	//kMeans.initializeCentroids()
	kMeans.centroidsIndexes = []int16{3, 5, 86, 109, 111, 243, 299, 375, 390, 475, 494, 538, 598, 670}

	kMeans.runSequentialClustering()
}

func weakScaling() {
	b, err := ioutil.ReadFile(
		"C:/Users/Nikola Zubic/Desktop/NTP projekat/K-MeansClusteringPythonGoPharo/Python/testData/College.csv")
	if err != nil {
		fmt.Print(err)
	}

	csvStr := string(b) // convert content to a 'string'

	df := dataframe.ReadCSV(strings.NewReader(csvStr)) // pass an io.Reader to the ReadCSV

	df = df.Select([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})

	kMeans := KMeansClustering{
		csvDf:                         df,
		sequentialGenerateResultsPath: "",
		parallelGenerateResultsPath:   "",
		k:                             14,
		maxIter:                       1000,
		centroidsIndexes:              nil,
	}

	//kMeans.initializeCentroids()
	kMeans.centroidsIndexes = []int16{3, 5, 86, 109, 111, 243, 299, 375, 390, 475, 494, 538, 598, 670}

	kMeans.runParallelClustering(14)
}

func strongScaling() {
	b, err := ioutil.ReadFile(
		"C:/Users/Nikola Zubic/Desktop/NTP projekat/K-MeansClusteringPythonGoPharo/Python/testData/College.csv")
	if err != nil {
		fmt.Print(err)
	}

	csvStr := string(b) // convert content to a 'string'

	df := dataframe.ReadCSV(strings.NewReader(csvStr)) // pass an io.Reader to the ReadCSV

	df = df.Select([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})

	for i := 2; i < 15; i++ {
		kMeans := KMeansClustering{
			csvDf:                         df,
			sequentialGenerateResultsPath: "",
			parallelGenerateResultsPath:   "",
			k:                             14,
			maxIter:                       1000,
			centroidsIndexes:              nil,
		}

		//kMeans.initializeCentroids()
		kMeans.centroidsIndexes = []int16{3, 5, 86, 109, 111, 243, 299, 375, 390, 475, 494, 538, 598, 670}

		kMeans.runParallelClustering(i)
	}

}

func runProgram() {
	/*
		go run kmeans.go "C:/Users/Nikola Zubic/Desktop/NTP projekat/K-MeansClusteringPythonGoPharo/Python/testData/College.csv" "C:/Users/Nikola Zubic/Desktop/results/firstExample/sequential" "C:/Users/Nikola Zubic/Desktop/results/firstExample/parallel" 3
	*/

	dataCsvPath := os.Args[1]
	sequentialResultsPathArg := os.Args[2]
	parallelResultsPathArg := os.Args[3]
	numberOfTasks := os.Args[4]

	numberOfTasksNum, err := strconv.Atoi(numberOfTasks)

	if err != nil {
		fmt.Println(err)
		return
	}

	b, err := ioutil.ReadFile(dataCsvPath)
	if err != nil {
		fmt.Print(err)
	}

	csvStr := string(b) // convert content to a 'string'

	df := dataframe.ReadCSV(strings.NewReader(csvStr)) // pass an io.Reader to the ReadCSV

	df = df.Select([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})

	kMeans := KMeansClustering{
		csvDf:                         df,
		sequentialGenerateResultsPath: sequentialResultsPathArg,
		parallelGenerateResultsPath:   parallelResultsPathArg,
		k:                             15,
		maxIter:                       100,
		centroidsIndexes:              nil,
	}

	kMeans.initializeCentroids()

	dataPointsSequential := kMeans.runSequentialClustering()
	dataPointsParallel := kMeans.runParallelClustering(numberOfTasksNum)

	checkIfEqual := mat.Equal(dataPointsSequential, dataPointsParallel)

	if !checkIfEqual {
		fmt.Println("[ERROR]: Algorithms aren't equal!")
	}
}

func main() {
	runProgram()

	/*
		sequentialClusteringExperiment()
		weakScaling()
		strongScaling()
	*/
}
