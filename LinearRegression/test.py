import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([[0, 3, 6],[1,2,3]]).T
ypoints = np.array([[0, 15, 25],[1,2,3]]).T

plt.plot(xpoints, ypoints)
plt.axis([0,10,0,50])
plt.show()