import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from non_stationnary_thermal.data_viewer import data_viewer


result = np.load("comparing/res_BC.npy")

dv = data_viewer(result, [10., 6., 205.], ["X (cm)", "Y(cm)", "T(s)"])
dv.show_fig()

X, t = np.meshgrid(np.linspace(0., 10., 201), np.linspace(0., 205., 41))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, t, result.mean(1).T)
plt.show()