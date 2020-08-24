import matplotlib.pyplot as plt

k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sequentialPython = [6.40445, 10.72678, 18.37369, 23.7978, 37.05416, 45.7364, 50.91486, 51.74426, 80.79497, 81.15289,
                 83.93705, 99.90131, 128.86062]
parallelPython = [1.42618, 9.74295, 15.48804, 17.24465, 26.38603, 29.13895, 35.65355, 37.27291, 52.17149, 73.14483,
                 92.42811, 92.65756, 214.28027]

plt.title("Python implementations: execution time comparison")
plt.xlabel("Number of clusters (=k) and number of tasks")
plt.ylabel("Execution time in seconds (=s)")
plt.plot(k, sequentialPython, 'r-')
for a, b in zip(k, sequentialPython):
    plt.text(a, b, str(b) + "s")
plt.plot(k, parallelPython, 'b-')
for a, b in zip(k, parallelPython):
    plt.text(a, b, str(b) + "s")
plt.legend(("Sequential", "Parallel"), loc='best')
plt.show()
