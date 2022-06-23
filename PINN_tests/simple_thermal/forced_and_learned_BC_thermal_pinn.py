import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""
loading distance and BC function
"""

inp_BC = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(5, activation="swish")(inp_BC)
outp_BC = tf.keras.layers.Dense(1, activation=None)(hid)

model_BC = tf.keras.Model(inp_BC, outp_BC)
model_BC.load_weights("W_BC.h5")

inp_dist = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(5, activation="swish")(inp_dist)
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
geometry
"""

# generate a grid geometry
L, l = 1.1, 0.4

L_lin = np.linspace(0., L, 200)
l_lin = np.linspace(0., l, 50)

X, Y = np.meshgrid(L_lin, l_lin)
vect = np.stack((X, Y), axis=-1)
vect = vect.reshape((-1, 2)) #grid as a vector list

vals = np.full(vect.shape[0], np.nan) #BC information storage
vals[vect[:, 0] == 0.] = 1200. #inlet values set at 1200°C
vals[vect[:, 0] == L] = 25. #outlet values set at 25°C 


test_values = (1175. / 1.1) * (1.1 - vect[:, 0]) + 25.

"""
pinn
"""

class PINN(tf.keras.Model):

    def train_step(self, data): #function called at each epoch
        inputs = data[0]
        vals = data[1]

        x = inputs[:, 0]
        y = inputs[:, 1]

        with tf.GradientTape() as gradtape:

            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x,y])
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch([x,y])
                    v = tf.stack([x,y], axis=1)
                    T = self(v, training=True) #neural network generate the predicted values
                T_x = tape2.gradient(T,x) #calculation of the first spacial derivatives
                T_y = tape2.gradient(T,y)
            T_xx = tape1.gradient(T_x,x) #calculation of the second spacial derivatives
            T_yy = tape1.gradient(T_y,y)

            res_T = T_xx + T_yy # thermal stationnary equation residuals
            loss = tf.math.reduce_mean(tf.math.abs(res_T)) #taking the mse only of the residuals

        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss}
            
    def test_step(self, data):
        inputs = data[0]
        vals = data[1]

        T_pred = self(inputs, training=False)
        err = self.compiled_loss(vals, T_pred, regularization_losses=self.losses) #computes the loss function between the known solution and the prediction
        return {"loss":err}

"""
Neural Network
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(20, activation="swish")(inp)
hid = tf.keras.layers.Dense(20, activation="swish")(hid)
hid = tf.keras.layers.Dense(20, activation="swish")(hid)
# hid = tf.keras.layers.Dense(40, activation="swish")(hid)
# hid = tf.keras.layers.Dense(40, activation="swish")(hid)
hid = tf.keras.layers.Dense(1, activation=None)(hid)
outp = BC_force_layer()(inp, hid)
#3 hidden layers of 20 neurons activated by ReLU function + 1 layer enforcing BCs


"""
main
"""

model = PINN(inp, outp)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
model.compile(optimizer=opt, loss="mse")

# model.run_eagerly = True
model.fit(vect, vals, epochs=500, batch_size=256, validation_data=(vect, test_values)) # 20 epochs but uses mini-batch, which is possible because only the local equation has to be solved
hist = model.history.history # logs metrics
model.save_weights("W_FaL_BC_thermal.h5")

"""
plot
"""

Ts = model.predict(vect)
Ts = Ts.reshape((50, 200)) # reformating model predicting

plt.imshow(Ts) # plotting result
plt.show()

for k in hist.keys(): # plotting all metrics
    met = hist[k]
    plt.plot(met, label=k)
plt.legend()
plt.show()

T_mean = np.mean(Ts, axis=0)
T_par = np.mean(model_BC.predict(vect).reshape((50, 200)), axis=0) # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
plt.plot(T_mean, label="result")
plt.plot(T_par, label="BC-enforcing function")
plt.legend()
plt.show()