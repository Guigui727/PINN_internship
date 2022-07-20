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
model_BC.load_weights("W_BC.h5") # loads 'particular' solution network

inp_dist = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp_dist)
hid = tf.keras.layers.Dense(30, activation="swish")(hid)
outp_dist = tf.keras.layers.Dense(1, activation=None)(hid)

model_distance = tf.keras.Model(inp_dist, outp_dist)
model_distance.load_weights("W_dist.h5") # loads distances network

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
geometry
"""

# generate a grid geometry


L, l, t = 10., 6., 205.

L_lin = np.linspace(0., L, 201)
l_lin = np.linspace(0., l, 3)
t_lin = np.linspace(0., t, 41)

L_s, l_s, t_s = L_lin.shape[0], l_lin.shape[0], t_lin.shape[0]

X = L_lin[:, np.newaxis].repeat(l_s, axis=1)[:, :, np.newaxis].repeat(t_s, axis=2)
Y = l_lin[np.newaxis, :].repeat(L_s, axis=0)[:, :, np.newaxis].repeat(t_s, axis=2)
Time = t_lin[np.newaxis, :].repeat(l_s, axis=0)[np.newaxis, :, :].repeat(L_s, axis=0) #reconstructing 3d data block
vect = np.stack((X, Y, Time), axis=-1)
vect = vect.reshape((-1, 3)) # grid as a vector list
vect = vect / np.array([L, L, t])[np.newaxis, :]
vect = vect.astype(np.float32)


"""
pinn
"""

class PINN(tf.keras.Model):
    def __init__(self, inp, outp, alpha=3818e-7):
        super(PINN, self).__init__(inp, outp)
        self.alpha = alpha


    def train_step(self, inputs): # function called at each epoch

        x = inputs[:, 0]
        y = inputs[:, 1]
        t = inputs[:, 2]

        with tf.GradientTape() as gradtape:

            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch([x, y, t])
                    v = tf.stack([x, y, t], axis=1)
                    T = self(v, training=True) # neural network generate the predicted values
                T_x = tape2.gradient(T,x) # calculation of the first spacial derivatives
                T_y = tape2.gradient(T,y)
                T_t = tape2.gradient(T,t)
            T_xx = tape1.gradient(T_x,x) # calculation of the second spacial derivatives
            T_yy = tape1.gradient(T_y,y)

            res_T = T_t - self.alpha * (T_xx + T_yy) # thermal stationnary equation residuals
            loss = tf.math.reduce_mean(tf.math.square(res_T)) # taking the mse only of the residuals

        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss}
            

"""
Neural Network
"""

inp = tf.keras.Input((3,))
hid = tf.keras.layers.Dense(40, activation="swish")(inp)
hid = tf.keras.layers.Dense(40, activation="swish")(hid)
hid = tf.keras.layers.Dense(40, activation="swish")(hid)
hid = tf.keras.layers.Dense(1, activation=None)(hid)
outp = BC_force_layer()(inp, hid)
#3 hidden layers of 40 neurons activated by Swish function + 1 layer enforcing BCs


"""
main
"""

alpha = 188e-3 * t / L**2 # in cm^2/s
model = PINN(inp, outp, alpha=alpha)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
model.compile(optimizer=opt, loss="mse")

epoch_per_loop = 20

for i in range(20):
    temporal_mask = vect[:, 2] <= (i + 1.) / 20.
    model.fit(vect[temporal_mask, :], epochs=epoch_per_loop * (i + 1), initial_epoch=epoch_per_loop * i, batch_size=512) # increases time slice every few epoch to allows the propagation of heat


hist = model.history.history # logs metrics
model.save_weights("W_FaL_BC_thermal.h5")

"""
plot
"""

Ts = model.predict(vect) * 1200.
Ts = Ts.reshape((L_s, l_s, t_s)) # reformating model predicting

T_par = 1200. * model_BC.predict(vect).reshape((L_s, l_s, t_s)).mean(axis=1) # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], Ts.mean(axis=1), label="NN", color='r')
ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], T_par, label="partial function", color='b') # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
plt.legend()
plt.show()

for k in hist.keys(): # plotting all metrics /!\ does not work with time slices now
    met = hist[k]
    plt.plot(met, label=k)
plt.legend()
plt.show()

view_BC = data_viewer(Ts, [L, l, t], ["X (cm)", "Y(cm)", "T(s)"])
view_BC.show_fig()
