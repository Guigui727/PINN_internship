import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


"""
geometry
"""

L, l = 1.1, 0.4

L_lin = np.linspace(0., L, 200)
l_lin = np.linspace(0., l, 50)

X, Y = np.meshgrid(L_lin, l_lin)
vect = np.stack((X, Y), axis=-1)
vect = vect.reshape((-1, 2)) #grid as a vector list


"""
dist pts
"""

id_bc = np.where(np.stack([vect[:, 0] == 0., vect[:, 0] == L]).any(axis=0))[0] #gets the indices of the boundary points
pts_bc = vect[id_bc, :] #gets the points

pts_bc_2 = np.repeat(pts_bc[:, np.newaxis, :], vect.shape[0], axis=1) #setups broadcast in order to calculate every distances between boundary points and every points
diff_squared = np.square(np.subtract(pts_bc_2, vect)) #computes every squared diff between values
dist = np.sqrt(diff_squared.sum(2)) #computes L2 norm
min_dist = dist.min(0) # gets minimal distance between a point and the boundaries
max_val = min_dist.max()
final_dist =  min_dist / max_val # "normalize" the distance between 0 and 1
final_dist_image = final_dist.reshape((50, 200)) # format result as a grid

"""
BC pts
"""
bc_val = np.zeros_like(pts_bc)
bc_val[pts_bc[:, 0] == 0.] = 1200.
bc_val[pts_bc[:, 0] == L] = 25. #associates to every boundary point to its value

"""
NN
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(5, activation="swish")(inp)
outp = tf.keras.layers.Dense(1, activation=None)(hid)
#simple NN with 1 hidden layer of 5 neurons activated by swish

"""
L-BFGS model
"""

class test_model(tf.keras.Model):
    def __init__(self, inp, outp):
        super(test_model, self).__init__(inp, outp)
        self.choice = True

        self.shapes = [tf.shape(x) for x in self.trainable_variables] #gets the shapes of trainable NN parameters
        self.flat_shapes = [tf.size(x) for x in self.trainable_variables] #gets the shapes flattened of trainable NN parameters
    

    def t_s_usual(self, data): #standard training
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {"loss": loss}


    def t_s_unusual(self, data): #training with L-BFGS
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

        return {"loss": loss, "converged":optim_res.converged}


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


model = test_model(inp, outp)
model.choice = False #selects L-BFGS
model.compile(loss="mse")

# model.run_eagerly = True
model.fit(vect, final_dist, epochs=40, batch_size = vect.shape[0] // 5 + 1)
hist = model.history.history # logs metrics
model.save_weights('W_dist.h5')


dist = model.predict(vect)
dist = dist.reshape((50, 200))

plt.imshow(dist) # plotting result
plt.show()
plt.imshow(dist - final_dist_image) # plotting result
plt.show()


met = hist["loss"]
plt.plot(met, label="loss")
plt.legend()
plt.show()

dist_mean = np.mean(dist, axis=0)
plt.plot(dist_mean, label='NN')
final_dist_mean = np.mean(final_dist_image, axis=0) # plotting the mean temperature along an axis to see if the solution is linear
plt.plot(final_dist_mean, label="true")
plt.legend()
plt.show()


"""
main BC
"""

model2 = test_model(inp, outp)
model2.choice = False
model2.compile(loss="mse")

model2.fit(pts_bc, bc_val, epochs=40, batch_size = pts_bc.shape[0] // 5 + 1)
hist = model2.history.history # logs metrics
model2.save_weights('W_BC.h5')


BC_inter = model2.predict(vect)
BC_inter = BC_inter.reshape((50, 200))

plt.imshow(BC_inter) # plotting result
plt.show()

met = hist["loss"]
plt.plot(met, label="loss")
plt.legend()
plt.show()

BC_mean = np.mean(BC_inter, axis=0)
plt.plot(BC_mean, label='NN')
plt.legend()
plt.show()