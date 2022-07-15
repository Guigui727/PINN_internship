import numpy as np
from scipy import signal
import scipy as scp

#for each dimension (min, max, nb points), nb points gives the size stored, but calculation are made on a bigger basis
x_val = (0., 0.1, 201)
y_val = (0., 0.06, 3)
t_val = (0., 205., 41)

ss = (5, 50000) #multiplier for spacial and temporal points number
slc = (slice(0, ss[0] * x_val[2] - 1, ss[0]), slice(0, ss[0] * y_val[2] - 1, ss[0])) #slices to sample stored points

result = np.zeros((x_val[2], y_val[2], t_val[2])) #stored result

T_tmp = np.full((x_val[2] * ss[0], y_val[2] * ss[0]), 25.) #working array, setup at 25°C
T_tmp[0, :] = 1200. # setting a border at 1200°C

result[:, :, 0] = T_tmp[slc[0], slc[1]] #storing ionit values

alpha = 188e-7 #thermal diffusivity (m^2/s)
beta = t_val[1] / (t_val[2] * ss[1]) # temporal resolution setup
gamma_x, gamma_y = x_val[1] / (x_val[2] * ss[0]), y_val[1] / (y_val[2] * ss[0]) #spacial resolution setup
gx_2, gy_2 = gamma_x ** -2, gamma_y ** -2

conv = (alpha * beta) * np.array([[0., gx_2, 0.], [gy_2, -2 * (gx_2 + gy_2), gy_2], [0., gx_2, 0.]]) # filter approximating delta_T for one spacial step

for i in range(1, t_val[2] * ss[1]):
    delta_T = scp.signal.convolve2d(T_tmp, conv, 'same', 'symm') # generating delta_T
    T_tmp += delta_T #updating the working array
    T_tmp[0, :] = 1200. #enforcing the boundary condition
    T_tmp[-1, :] = 25.

    if i % ss[1] == 0:
        result[:, :, i // ss[1]] = T_tmp[slc[0], slc[1]] # stores the values after a few iterations
        print("{:.2f}%".format(100 * i / (ss[1] * t_val[2])))

np.save("res.npy", result)
