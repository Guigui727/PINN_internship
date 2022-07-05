import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_viewer import data_viewer
from geo_gen import geo_gen
from scipy.interpolate import griddata, RBFInterpolator

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
L_s, l_s, t_s = 100, 25, 200

X, Y, Time = np.meshgrid(
    np.linspace(0., L, 200),
    np.linspace(0., l,50),
    np.linspace(0., t, 200),
    indexing='ij'
)
vect, angles, phy_BC_ids, ang_ids = geo_gen()
vect = vect.astype(np.float32) / np.array([L, L, t], dtype=np.float32)

"""
block making
"""

def as_block(vectors, values, ss=(200, 50, 200)):
    xr, yr, r = 0.2 / L, 0.2 / L, 0.05 / L

    x_off, y_off, z_off = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    x_on, y_on, z_on = np.meshgrid(
        np.linspace(x_off.min(), x_off.max(), ss[0]),
        np.linspace(y_off.min(), y_off.max(), ss[1]),
        np.linspace(z_off.min(), z_off.max(), ss[2]),
        indexing='ij'
    )
    # flat_vec = np.stack([x_on.ravel(), y_on.ravel(), z_on.ravel()], axis=-1)

    block = griddata(vectors, values, (x_on, y_on, z_on), method='nearest').squeeze()
    # interp = RBFInterpolator(vectors, values, neighbors=15, kernel='linear')
    # block = interp(flat_vec)
    # block = block.reshape(ss)

    dist = (x_on - xr)**2 + (y_on - yr)**2
    mask = dist <= r**2
    block[mask] = np.nan

    return block

"""
plotting
"""

model = tf.keras.Model(inp, outp)
model.load_weights("W_FaL_BC_thermal.h5")

Ts = model.predict(vect) * 1200.
Ts = as_block(vect, Ts) # reformating model predicting

T_par = 1200. * np.nanmean(as_block(vect, model_BC.predict(vect)), axis=1) # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], np.nanmean(Ts, axis=1), label="NN", color='r')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], T_par, label="partial function", color='b') # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
plt.legend()
plt.show()

view_BC = data_viewer(Ts, [L, l, t], ["X (m)", "Y(m)", "T(s)"])
view_BC.show_fig()
