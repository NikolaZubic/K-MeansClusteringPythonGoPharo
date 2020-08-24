import matplotlib.pyplot as plt

k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
sequentialGo = [0.01896, 0.02591, 0.04089, 0.04488, 0.05086, 0.05984, 0.08378, 0.09577, 0.09859, 0.13666, 0.16054,
                0.18553, 0.21045]
parallelGo = [0.01597, 0.01992, 0.02693, 0.02892, 0.03687, 0.04189, 0.04985, 0.05784, 0.05785, 0.08577, 0.08776,
              0.09872, 0.1027]

plt.title("Go(lang) implementations: execution time comparison")
plt.xlabel("Number of clusters (=k) and number of tasks")
plt.ylabel("Execution time in seconds (=s)")
plt.plot(k, sequentialGo, 'r-')
for a, b in zip(k, sequentialGo):
    plt.text(a, b, str(b) + "s")
plt.plot(k, parallelGo, 'b-')
for a, b in zip(k, parallelGo):
    plt.text(a, b, str(b) + "s")
plt.legend(("Sequential", "Parallel"), loc='best')
plt.show()
