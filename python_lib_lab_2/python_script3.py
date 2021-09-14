import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-3, 3, 51)
y1 = x ** 2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)
fig, ax = plt.subplots()
plt.figure(figsize=(8, 6))
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.xticks([-5, 5])
plt.show()
