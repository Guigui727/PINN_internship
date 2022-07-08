import numpy as np
import matplotlib.pyplot as plt
from data_viewer import data_viewer

result = np.load("res.npy")

dv = data_viewer(result, [1.1, 0.4, 18000.], ["X (m)", "Y(m)", "T(s)"])
dv.show_fig()

X, t = np.meshgrid(np.linspace(0., 1.1, 100), np.linspace(0., 18000., 200))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, t, result.mean(1).T)
plt.show()