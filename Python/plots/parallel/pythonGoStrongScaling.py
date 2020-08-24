import matplotlib.pyplot as plt

# k is fixed to be 14
"""
Starting strong parallel clustering experiment...
74.93615 seconds
3.24402 seconds
1.70304 seconds
1.73539 seconds
1.95274 seconds
2.22658 seconds
2.37868 seconds
2.58508 seconds
2.92018 seconds
3.22238 seconds
91.57144 seconds
123.77689 seconds
214.28027 seconds
Strong parallel clustering experiment finished!

Golang:
0.10175 seconds
0.08874 seconds
0.08677 seconds
0.09175 seconds
0.08777 seconds
0.08477 seconds
0.08178 seconds
0.08477 seconds
0.08378 seconds
0.08378 seconds
0.08577 seconds
0.08278 seconds
0.18577 seconds
"""

number_of_tasks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
pythonSeconds = [74.93615, 3.24402, 1.70304, 1.73539, 1.95274, 2.22658, 2.37868, 2.58508, 2.92018, 3.22238, 91.57144,
                 123.77689, 214.28027]
goSeconds = [0.10175, 0.08874, 0.08677, 0.09175, 0.08777, 0.08477, 0.08178, 0.08477, 0.08378, 0.08378, 0.08577, 0.08278,
             0.18577]

plt.title("Execution time comparison between parallel implementations (strong scaling), k=14")
plt.xlabel("Number of tasks")
plt.ylabel("Execution time in seconds (=s)")
plt.plot(number_of_tasks, pythonSeconds, 'r-')
for a, b in zip(number_of_tasks, pythonSeconds):
    plt.text(a, b, str(b) + "s")
plt.plot(number_of_tasks, goSeconds, 'b-')
for a, b in zip(number_of_tasks, goSeconds):
    plt.text(a, b, str(b) + "s")
plt.legend(("Python", "Go(lang)"), loc='best')
plt.show()
