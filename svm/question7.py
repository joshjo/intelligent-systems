import numpy as np
import matplotlib.pyplot as plt


ax = plt.axes()
sqrt_05 = np.sqrt(0.5)

plt.scatter([1, 4, 4, 2, 3], [6, 9, 6, 2, 1], marker='x')
plt.scatter([5, 9, 9, 0], [1, 5, 1, 3], marker='o')

x = np.linspace(0, 10)

ax.plot(x, x - 1, label=r'$h_1$')

plt.legend()
plt.show()
    