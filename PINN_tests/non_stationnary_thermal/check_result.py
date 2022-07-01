import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_viewer import data_viewer


"""
loading distance and BC function
"""

inp_BC = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp_BC)
hid = tf.keras.layers.Dense(30, activation="swish")(hid)
outp_BC = tf.keras.layers.Dense(1, activation=None)(hid)

model_BC = tf.keras.Model(inp_BC, outp_BC)
model_BC.load_weights("W_BC.h5")

inp_dist = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp_dist)
hid = tf.keras.layers.Dense(30, activation="swish")(hid)
outp_dist = tf.keras.layers.Dense(1, activation=None)(hid)

model_distance = tf.keras.Model(inp_dist, outp_dist)
model_distance.load_weights("W_dist.h5")


"""
layer enforcing BCs
"""

class BC_force_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(BC_force_layer, self).__init__()


    def call(self, inputs, layer):
        geo_lim = model_distance(inputs)
        BC_fill = model_BC(inputs)
        result = tf.math.add(
            BC_fill,
            tf.math.multiply(layer, geo_lim) # ponders the NN output by the distance function and adds the BC-interpolating function
        )
        return result

"""
Neural Network
"""

inp = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp)
hid = tf.keras.layers.Dense(40, activation="swish")(hid)
hid = tf.keras.layers.Dense(40, activation="swish")(hid)
hid = tf.keras.layers.Dense(1, activation=None)(hid)
outp = BC_force_layer()(inp, hid)
#3 hidden layers of 20 neurons activated by Swish function + 1 layer enforcing BCs


"""
geometry
"""

L, l, t = 1.1, 0.4, 1000.

L_lin = np.linspace(0., L, 100)
l_lin = np.linspace(0., l, 25)
t_lin = np.linspace(0., t, 200)

L_s, l_s, t_s = L_lin.shape[0], l_lin.shape[0], t_lin.shape[0]

X = L_lin[:, np.newaxis].repeat(l_s, axis=1)[:, :, np.newaxis].repeat(t_s, axis=2)
Y = l_lin[np.newaxis, :].repeat(L_s, axis=0)[:, :, np.newaxis].repeat(t_s, axis=2)
Time = t_lin[np.newaxis, :].repeat(l_s, axis=0)[np.newaxis, :, :].repeat(L_s, axis=0)
vect = np.stack((X, Y, Time), axis=-1)
vect = vect.reshape((-1, 3)) #grid as a vector list
vect = vect / np.array([L, L, t])[np.newaxis, :]
vect = vect.astype(np.float32)

"""
plotting
"""

model = tf.keras.Model(inp, outp)
model.load_weights("W_FaL_BC_thermal.h5")

Ts = model.predict(vect) * 1200.
Ts = Ts.reshape((L_s, l_s, t_s)) # reformating model predicting

T_par = 1200. * model_BC.predict(vect).reshape((L_s, l_s, t_s)).mean(axis=1) # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], Ts.mean(axis=1), label="NN", color='r')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], T_par, label="partial function", color='b') # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
plt.legend()
plt.show()

view_BC = data_viewer(Ts, [L, l, t], ["X (m)", "Y(m)", "T(s)"])
view_BC.show_fig()
