import matplotlib.pyplot as plt

k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

pythonSeconds = [1.42618, 9.74295, 15.48804, 17.24465, 26.38603, 29.13895, 35.65355, 37.27291, 52.17149, 73.14483,
                 92.42811, 92.65756, 214.28027]
goSeconds = [0.01597, 0.01992, 0.02693, 0.02892, 0.03687, 0.04189, 0.04985, 0.05784, 0.05785, 0.08577, 0.08776, 0.09872,
             0.1027]

plt.title("Execution time comparison between parallel implementations (weak scaling)")
plt.xlabel("Number of clusters (=k)")
plt.ylabel("Execution time in seconds (=s)")
plt.plot(k, pythonSeconds, 'r-')
for a, b in zip(k, pythonSeconds):
    plt.text(a, b, str(b) + "s")
plt.plot(k, goSeconds, 'b-')
for a, b in zip(k, goSeconds):
    plt.text(a, b, str(b) + "s")
plt.legend(("Python", "Go(lang)"), loc='best')
plt.show()
