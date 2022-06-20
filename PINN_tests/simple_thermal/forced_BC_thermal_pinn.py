import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""
layer enforcing BCs
"""

class BC_force_layer(tf.keras.layers.Layer):
    def __init__(self, L, temp_cold, temp_hot):
        super(BC_force_layer, self).__init__()
        self.L, self.temp_cold, self.temp_hot = L, temp_cold, temp_hot

    @tf.function
    def geo_func(self, Xs):
        return tf.math.multiply(Xs, tf.math.subtract(self.L, Xs)) #smooth distance function =0 for x=0 or x=L

    @tf.function
    def BC_filler(self, Xs):
        return tf.math.add(
            tf.math.scalar_mul(tf.constant(1175./1.21), tf.math.square(Xs)), #function interpolating all the BCs (volontarily non linear to force the network to learn something)
            tf.constant(25.)
        )

    def call(self, inputs, layer):
        Xs = inputs[:, 0]
        BC_fill = self.BC_filler(Xs)[:, tf.newaxis]
        geo_lim = self.geo_func(Xs)[:, tf.newaxis]
        result = tf.math.add(
            BC_fill,
            tf.math.multiply(layer, geo_lim) # ponders the NN output by the distnace function and adds the BC-interpolating function
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
            loss = tf.math.reduce_mean(tf.math.square(res_T)) #taking the mse only of the residuals

        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss}

"""
Neural Network
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(20, activation="relu")(inp)
hid = tf.keras.layers.Dense(20, activation="relu")(hid)
hid = tf.keras.layers.Dense(20, activation="relu")(hid)
hid = tf.keras.layers.Dense(1, activation="linear")(hid)
outp = BC_force_layer(L, 1200., 25.)(inp, hid)
#3 hidden layers of 20 neurons activated by ReLU function + 1 layer enforcing BCs


"""
main
"""

model = PINN(inp, outp)
opt = tf.keras.optimizers.Adam(learning_rate=2e-3)
model.compile(optimizer=opt)

# model.run_eagerly = True
model.fit(vect, vals, epochs=20) # 20 epochs but uses mini-batch, which is possible because only the local equation has to be solved
hist = model.history.history # logs metrics

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
T_par = (1175./1.21)* np.linspace(0., 1.1, 200)**2 + 25. # plotting mean along an axis and the BC-interpolating function to see if the NN as learn correctly the solution
plt.plot(T_mean, label="result")
plt.plot(T_par, label="BC-enforcing function")
plt.legend()
plt.show()