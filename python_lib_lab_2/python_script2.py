import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 10, 51)
f = np.cos(t)
plt.plot(t, f, color='g')
plt.xticks([0.5, 9.5])
plt.yticks([-2.5, 2.5])
plt.title('График f(t)')
plt.show()

