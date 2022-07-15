import numpy as np
import matplotlib.pyplot as plt
from data_viewer import data_viewer

result = np.load("res.npy")

dv = data_viewer(result, [0.1, 0.06, 205.], ["X (m)", "Y(m)", "T(s)"])
dv.show_fig()

X, t = np.meshgrid(np.linspace(0., 0.1, 201), np.linspace(0., 205., 41))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, t, result.mean(1).T)
plt.show()