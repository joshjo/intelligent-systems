import numpy as np
import matplotlib.pyplot as plt


ax = plt.axes()
sqrt_05 = np.sqrt(0.5)

plt.scatter([1, 4, 4], [6, 9, 6], marker='x')
plt.scatter([5, 9, 9], [1, 5, 1], marker='o')

x = np.linspace(0, 10)

ax.plot(x, x - 1, label=r'$h_1$')
ax.plot(x, (2*x+32)/7, label=r'$h_2$')
ax.plot(x, (sqrt_05*x-sqrt_05)/sqrt_05, label=r'$h_3$')
ax.plot(x, (2*x-32)/7, label=r'$h_4$')
ax.plot(x, x - 4, label=r'$hx_1$')

plt.legend()
plt.show()
    