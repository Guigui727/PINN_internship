import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from data_viewer import data_viewer

import os
files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(os.getcwd(), f))]


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
Time = t_lin[np.newaxis, :].repeat(l_s, axis=0)[np.newaxis, :, :].repeat(L_s, axis=0) #reconstructing 3d data block
vect = np.stack((X, Y, Time), axis=-1)
vect = vect.reshape((-1, 3)) #grid as a vector list
vect = vect / np.array([L, L, t])[np.newaxis, :]
vect = vect.astype(np.float32) # as tensorflows deals with float 32, it is set here to shorten following calculations


"""
dist pts
"""

id_bc = np.where(np.stack([vect[:, 0] == 0., vect[:, 0] == 1., vect[:, 2] == 0.]).any(axis=0))[0] #gets the indices of the boundary points (x=0 | x=L | t=0)
pts_bc = vect[id_bc, :] #gets the points

def gen_dist(batch_slices): #in a function to wipe unnecessary variables

    list_slice = [list(range(i * batch_slices, (i + 1) * batch_slices)) for i in range (vect.shape[0] // batch_slices)] #cannot fully parallelize due to data size, compromising with a batched approach
    if vect.shape[0] % batch_slices != 0:
        last_id = list_slice[-1][-1]
        list_slice.append(list(range(last_id + 1, vect.shape[0])))

    min_dist = []
    pts_bc_2 = np.repeat(pts_bc[:, np.newaxis, :], batch_slices, axis=1) #setups broadcast in order to calculate every distances between boundary points and every points
    for i in range(len(list_slice)):  
        b = list_slice[i]
        diff_squared = np.square(np.subtract(pts_bc_2[:, :len(b), :], vect[b, :])) #computes every squared diff between values
        dist = np.sqrt(diff_squared.sum(2)) #computes L2 norm
        min_dist.append(dist.min(0)) # gets minimal distance between a point and the boundaries
        print("{} done out of {}".format(i + 1, len(list_slice)))

    min_dist = np.concatenate(min_dist, axis=0)
    max_val = min_dist.max()
    return min_dist / max_val # "normalize" the distance between 0 and 1

if "pregenerated_data.npy" in files:
    final_dist = np.load("pregenerated_data.npy") #avoid to re-compute min-distances each time /!\ Delete the file if input data changes !!
else:
    final_dist = gen_dist(12000)
    np.save("pregenerated_data.npy", final_dist)
final_dist_block = final_dist.reshape((L_s, l_s, t_s))

"""
BC pts
"""
bc_val = np.full(pts_bc.shape[0], 25., dtype=np.float32)
msk = pts_bc[:, 0] == 0.
coef = 2e1
bc_val[msk] = 25. + 1175. * (1. - np.exp(- coef * pts_bc[msk, 2]))#associates to every boundary point to its value
bc_val = bc_val / 1200. # divides by a reference value

"""
NN
"""
def create_NN():
    inp = tf.keras.Input((3,))
    hid = tf.keras.layers.Dense(40, activation="swish")(inp)
    hid = tf.keras.layers.Dense(30, activation="swish")(hid)
    outp = tf.keras.layers.Dense(1, activation=None)(hid)
    #simple NN with 2 hidden layer of 40 and 30 neurons activated by swish
    return inp, outp

"""
L-BFGS model (NOT USED HERE)
"""

class LBFGS_model(tf.keras.Model):
    def __init__(self, inp, outp):
        super(LBFGS_model, self).__init__(inp, outp)

        self.shapes = [tf.shape(x) for x in self.trainable_variables] #gets the shapes of trainable NN parameters
        self.flat_shapes = [tf.size(x) for x in self.trainable_variables] #gets the shapes flattened of trainable NN parameters
    
    def train_step(self, data):
        x, y = data
        
        def fun(train_var): 
            for a, b in zip(self.trainable_variables, self.reshaping(train_var)):
                a.assign(b)
            return self.compiled_loss(y, self(x, training=True), regularization_losses=self.losses) #return the loss function to optimize

        def fun_val_and_grad(train_var): #return the loss function and its gradient
            with tf.GradientTape() as tape:
                loss = fun(train_var)
            grads = tape.gradient(loss, self.trainable_weights)
            r = (loss, self.flattening(grads))
            return r

        optim_res = tfp.optimizer.lbfgs_minimize(
            fun_val_and_grad,
            initial_position = self.flattening(self.trainable_variables),
            max_iterations = 50
        ) #tensorflow probability implementation of L-BFGS
        
        for a, b in zip(self.trainable_variables, self.reshaping(optim_res.position)):
                a.assign(b)
        loss = self.compiled_loss(y, self(x, training=True), regularization_losses=self.losses)

        return {"loss": loss, "converged":optim_res.converged, "failed":optim_res.failed}


    def flattening(self, vect_list): #flatten the trainable variables values 
        l = []
        for v in vect_list:
            l.append(tf.reshape(v, (-1,)))
        r = tf.concat(l, axis=0)
        return r


    def reshaping(self, vect): #reshapes a flat vector representing all the NN parameters to all the trainable varables
        flat_list = tf.split(vect, self.flat_shapes)
        return [tf.reshape(fv, fvs) for fv, fvs in zip(flat_list, self.shapes)]



"""
main distance
"""

def main_dist(plotting=True, training=True):
    model = tf.keras.Model(*create_NN())
    if training:
        weight_list = np.full((vect.shape[0]), 1.)
        weight_list[id_bc] = 2 * (vect.shape[0] - id_bc.shape[0]) / id_bc.shape[0]
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
        model.compile(optimizer=opt, loss="mse")

        model.fit(vect, final_dist, epochs=5, batch_size = vect.shape[0] // 3999, sample_weight=weight_list)
        hist = model.history.history # logs metrics
        model.save_weights('W_dist.h5')
    else:
        model.load_weights('W_dist.h5')

    if plotting:
        dist = model.predict(vect)
        dist = dist.reshape((L_s, l_s, t_s))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], dist.mean(1), label="NN", color='r')
        ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], final_dist_block[:, 0, :], label="calculated distance", color='b')
        plt.legend()
        plt.show()

        if training:
            met = hist["loss"]
            plt.plot(met, label="loss")
            plt.legend()
            plt.show()

        view_dist = data_viewer(dist, [L, l, t], ["X (m)", "Y(m)", "T(s)"])
        view_dist.show_fig()

main_dist()

"""
main BC
"""

def main_BC(plotting=True, training=True):
    model2 = tf.keras.Model(*create_NN())
    if training:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
        model2.compile(optimizer=opt, loss="mse")

        model2.fit(pts_bc, bc_val, epochs=20, batch_size = pts_bc.shape[0] // 3000)
        hist = model2.history.history # logs metrics
        model2.save_weights('W_BC.h5')
    else:
        model2.load_weights('W_BC.h5')

    if plotting:
        BC_inter = model2.predict(vect) * 1200.
        BC_inter = BC_inter.reshape((L_s, l_s, t_s))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(X[:, 0, :], Time[:, 0, :], BC_inter.mean(1), label="NN")
        plt.legend()
        plt.show()

        if training:
            met = hist["loss"]
            plt.plot(met, label="loss")
            plt.legend()
            plt.show()

        view_BC = data_viewer(BC_inter, [L, l, t], ["X (m)", "Y(m)", "T(s)"])
        view_BC.show_fig()

main_BC()