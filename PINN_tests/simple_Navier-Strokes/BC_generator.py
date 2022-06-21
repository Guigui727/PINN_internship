import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import geometry_generator as geogen
from scipy.interpolate import griddata


"""
geometry
"""

# generate a grid geometry
L, l = 1.1, 0.4
Umax = 1.

tool, surround = geogen.tool() 
tool, surround = tool + np.array([[0.2, 0.2]]), surround + np.array([[0.2, 0.2]])
body = geogen.body(0.1)

vect = np.concatenate([tool, surround, body], axis=0)

vals = np.full((vect.shape[0], 3), np.nan) #BC information storage
inlet_points = vect[:, 0] == 0.
vals[inlet_points, 1] = 0. #v(0,y)=0
vals[inlet_points, 0] =  4 * Umax * (l - vect[inlet_points, 1]) * vect[inlet_points, 1] / l**2
wall_points = np.stack([vect[:, 1] == 0., vect[:, 1] == L]).any(axis=0)
vals[wall_points, 0:2] = np.array([0., 0.])
outlet_points = vect[:, 0] == L
vals[outlet_points, 2] = 0.

"""
BC gen
"""

class BC_generator(tf.keras.Model):


    def train_step(self, data): #function called at each epoch
        inputs = data[0]
        vals = data[1]

        mask = tf.math.is_nan(vals)
        masks = tf.unstack(mask, axis=1)
        ids_BC = [tf.where(m) for m in masks]
        ids_non_BC = [tf.where(tf.math.logical_not(m)) for m in masks]

        with tf.GradientTape() as gradtape:       
            generated_outputs = self(inputs, training=True) #neural network generate the predicted values
            BC_func, distance = tf.unstack(generated_outputs, axis=2)


            BC_pts = [tf.gather(BC_func[:, i], ids_BC[i]) for i in range(len(ids_BC))]
            BC_vals = [tf.gather(vals[:, i], ids_BC[i]) for i in range(len(ids_BC))]
            BC_dist = [tf.gather(distance[:, i], ids_BC[i]) for i in range(len(ids_BC))]
            non_BC_dist = [tf.gather(distance[:, i], ids_non_BC[i]) for i in range(len(ids_non_BC))]

            loss_BC_func = tf.math.reduce_sum(tf.stack([
                tf.math.reduce_mean(tf.math.squared_difference(i, j)) for (i, j) in zip(BC_pts, BC_vals)
            ], axis=-1))

            loss_distance = tf.math.add(
                tf.math.reduce_sum(tf.stack([
                    tf.math.reduce_mean(tf.math.square(i)) for i in BC_dist
                ], axis=-1)),
                tf.math.reduce(tf.stack([
                    tf.math.reduce_mean(tf.math.square(tf.gather(i, tf.where(tf.math.less(i, tf.constant(0.)))))) for i in non_BC_dist
                ], axis=-1))
            )

            

            loss = tf.math.add(loss_BC_func, loss_distance) #summing the mse
        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss, "loss_BC":loss_BC_func, "loss_distance":loss_distance}

"""
Neural Network
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(30, activation="relu")(inp)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(6, activation="linear")(hid)
outputs = tf.keras.layers.Reshape((-1, 3, 2))(hid)
