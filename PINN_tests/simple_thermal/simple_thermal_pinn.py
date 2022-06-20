import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
            loss_res = tf.math.reduce_mean(tf.math.square(res_T)) #taking the mse of the residuals

            loss_inlet = tf.math.reduce_mean(tf.math.square(tf.math.subtract(
                T[vals == 1200.], # taking the inlet points to inforce a temperature 
                tf.constant(1200.)
            )))
            loss_outlet = tf.math.reduce_mean(tf.math.square(tf.math.subtract(
                T[vals == 25.], # taking the outlet points to inforce a temperature 
                tf.constant(25.)
            )))
            loss = loss_res + loss_inlet + loss_outlet #summing the mse
        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss, "loss_res":loss_res, "loss_inlet":loss_inlet, "loss_outlet":loss_outlet}

"""
Neural Network
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(20, activation="relu")(inp)
hid = tf.keras.layers.Dense(20, activation="relu")(hid)
hid = tf.keras.layers.Dense(20, activation="relu")(hid)
outp = tf.keras.layers.Dense(1, activation="linear")(hid)
#3 hidden layers of 20 neurons activated by ReLU function


"""
main
"""

model = PINN(inp, outp)
opt = tf.keras.optimizers.Adam(learning_rate=2e-3)
model.compile(optimizer=opt)

model.fit(vect, vals, epochs=3000, batch_size=10000) #full-batch because if minibatch is used, one minibatch could have no inlet/wall/outlet points and generated a nan value
hist = model.history.history # logs metrics

"""
plot
"""

Ts = model.predict(vect)
Ts = Ts.reshape((50, 200)) #reformating model predicting

plt.imshow(Ts) # plotting result
plt.show()

for k in hist.keys(): # plotting all metrics
    met = hist[k]
    plt.plot(met, label=k)
plt.legend()
plt.show()

T_mean = np.mean(Ts, axis=0) # plotting the mean temperature along an axis to see if the solution is linear
plt.plot(T_mean)
plt.show()