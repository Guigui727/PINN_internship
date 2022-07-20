import numpy as np
from scipy import signal
import scipy as scp

#for each dimension (min, max, nb points), nb points gives the size stored, but calculation are made on a bigger basis
x_val = (0., 10., 201)
y_val = (0., 6., 3)
t_val = (0., 205., 41)
alpha = 188e-3 #thermal diffusivity (cm^2/s)

sst =  10000 #multiplier for temporal points number


k = (t_val[1] - t_val[0]) / (t_val[2] * sst)
h = (6. * alpha * k) ** 0.5
ssx = (x_val[1] - x_val[0]) / (x_val[2] * h)
ssy = (y_val[1] - y_val[0]) / (y_val[2] * h)
ss = (int(np.floor(ssx)), int(np.floor(ssy)), sst)


slc = (slice(0, ss[0] * x_val[2] - 1, ss[0]), slice(0, ss[1] * y_val[2] - 1, ss[1])) #slices to sample stored points

result = np.zeros((x_val[2], y_val[2], t_val[2])) #stored result

T_tmp = np.full((x_val[2] * ss[0], y_val[2] * ss[1]), 25.) #working array, setup at 25°C

result[:, :, 0] = T_tmp[slc[0], slc[1]] #storing init values

conv = ( 1. / 6.) * np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]]) # convolutionnal kernel applying one temporal step to 

j = 0
incr = 1200. / float(ss[2])
for i in range(1, t_val[2] * ss[2]):
    T_tmp = scp.signal.convolve2d(T_tmp, conv, 'same', 'symm') #updating the working array
    T_tmp[0, :] = np.min([1200., incr * i]) #enforcing the boundary condition
    T_tmp[-1, :] = 25.

    if i % ss[2] == 0:
        result[:, :, i // ss[2]] = T_tmp[slc[0], slc[1]] # stores the values after a few iterations
        print("{:.2f}%".format(100 * i / (ss[2] * t_val[2])))

np.save("res.npy", result)
