import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


ax = plt.axes()
sqrt_05 = np.sqrt(0.5)

y0 = [[1, 6], [4, 9], [4, 6], [2, 2], [3, 1]]
y1 = [[5, 1], [9, 5], [9, 1], [0, 3]]

plt.scatter(*zip(*y0), marker='x')
plt.scatter(*zip(*y1), marker='o')

x = np.linspace(0, 10)

ax.plot(x, x - 1, label=r'$h_1$')


def rect(x):
    return x[0] - x[1] - 1


def dist(point):
    return abs(rect(x)) / LA.norm(point, 2)


if __name__ == '__main__':
    print([rect(i) for i in y0])
    print([rect(i) for i in y1])
    print(
        [rect(i) for i in [[2, 2], [3, 1], [0, 3]]])
    # plt.show()
