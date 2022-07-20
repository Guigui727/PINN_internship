import numpy as np
import matplotlib.pyplot as plt

result = np.load("res.npy")
result = result.mean(1)
abaqus = np.load("abaqus.npy")

im = plt.imshow((result - abaqus).T, extent=[0., 0.1, 205., 0.], aspect='auto')
plt.colorbar(im, orientation='horizontal', label='\u0394T (°C)')
plt.xlabel('x (m)')
plt.ylabel('t (s)')
plt.show()

im2 = plt.imshow(np.abs((result - abaqus).T / abaqus.T), extent=[0., 0.1, 205., 0.], aspect='auto')
plt.colorbar(im2, orientation='horizontal', label='absolute relative error (%)')
plt.xlabel('x (m)')
plt.ylabel('t (s)')
plt.show()

