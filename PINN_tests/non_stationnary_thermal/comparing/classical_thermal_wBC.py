import numpy as np
from scipy import signal
import scipy as scp
import tensorflow as tf

import sys
sys.path.append('../')

"""
getting BC
"""

inp_BC = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp_BC)
hid = tf.keras.layers.Dense(30, activation="swish")(hid)
outp_BC = tf.keras.layers.Dense(1, activation=None)(hid)

model_BC = tf.keras.Model(inp_BC, outp_BC)
model_BC.load_weights("W_BC.h5")


"""
simulation
"""

#for each dimension (min, max, nb points), nb points gives the size stored, but calculation are made on a bigger basis
x_val = (0., 10., 201)
y_val = (0., 6., 3)
t_val = (0., 205., 41)
alpha = 188e-3 #thermal diffusivity (cm^2/s)

sst =  10000 #multiplier for temporal points number

# setting h,k to get a simple kernel
k = (t_val[1] - t_val[0]) / (t_val[2] * sst)
h = (6. * alpha * k) ** 0.5
ssx = (x_val[1] - x_val[0]) / (x_val[2] * h)
ssy = (y_val[1] - y_val[0]) / (y_val[2] * h)
ss = (int(np.floor(ssx)), int(np.floor(ssy)), sst)


slc = (slice(0, ss[0] * x_val[2] - 1, ss[0]), slice(0, ss[1] * y_val[2] - 1, ss[1])) #slices to sample stored points

result = np.zeros((x_val[2], y_val[2], t_val[2])) #stored result

#getting only init points
vect_T_init = np.stack((np.meshgrid(np.linspace(x_val[0], x_val[1], x_val[2] * ss[0]), np.linspace(y_val[0], y_val[1], y_val[2] * ss[1]), np.array([0.]), indexing='ij')), axis=-1).reshape((-1,3))  / np.array([x_val[1], x_val[1], t_val[1]])[np.newaxis, :]

T_tmp = 1200. * model_BC.predict(vect_T_init).reshape((x_val[2] * ss[0], y_val[2] * ss[1])) #working array, setup like the boundary NN

del vect_T_init

result[:, :, 0] = T_tmp[slc[0], slc[1]] #storing init values

conv = ( 1. / 6.) * np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]]) # convolutionnal kernel applying one temporal step to 

#getting hot and cold side points. They are splitted because it would be too big of an array otherwise.
vect_T_hot = [v.reshape((-1, 3)) / np.array([x_val[1], x_val[1], t_val[1]])[np.newaxis, :] for v in np.split(np.stack((np.meshgrid(np.array([x_val[0]]), np.linspace(y_val[0], y_val[1], y_val[2] * ss[1]), np.linspace(t_val[0], t_val[1], t_val[2] * ss[2]), indexing='ij')), axis=-1), t_val[2], axis = 2)]
vect_T_cold = [v.reshape((-1, 3)) / np.array([x_val[1], x_val[1], t_val[1]])[np.newaxis, :] for v in np.split(np.stack((np.meshgrid(np.array([x_val[1]]), np.linspace(y_val[0], y_val[1], y_val[2] * ss[1]), np.linspace(t_val[0], t_val[1], t_val[2] * ss[2]), indexing='ij')), axis=-1), t_val[2], axis = 2)]

T_hot = 1200. * model_BC.predict(vect_T_hot[0], batch_size=8192, verbose=1).reshape((y_val[2] * ss[1], ss[2]))
T_cold = 1200. * model_BC.predict(vect_T_cold[0], batch_size=8192, verbose=1).reshape((y_val[2] * ss[1], ss[2]))


for i in range(1, t_val[2] * ss[2]):
    T_tmp = scp.signal.convolve2d(T_tmp, conv, 'same', 'symm') #updating the working array
    T_tmp[0, :] = T_hot[:, i % t_val[2]] #enforcing the boundary condition
    T_tmp[-1, :] = T_cold[:, i % t_val[2]]

    if i % ss[2] == 0:
        result[:, :, i // ss[2]] = T_tmp[slc[0], slc[1]] # stores the values after a few iterations
        print("{:.2f}%".format(100 * i / (ss[2] * t_val[2])))

        del T_hot, T_cold
        T_hot = 1200. * model_BC.predict(vect_T_hot[i // ss[2]], batch_size=8192, verbose=1).reshape((y_val[2] * ss[1], ss[2]))
        T_cold = 1200. * model_BC.predict(vect_T_cold[i // ss[2]], batch_size=8192, verbose=1).reshape((y_val[2] * ss[1], ss[2]))

np.save("comparing/res_BC.npy", result)
