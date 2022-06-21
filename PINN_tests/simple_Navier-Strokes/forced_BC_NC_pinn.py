import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import geometry_generator as geogen
from scipy.interpolate import griddata


class BC_force_layer(tf.keras.layers.Layer):
    def __init__(self, L, l):
        super(BC_force_layer, self).__init__()
        self.L, self.l = L, l

    @tf.function
    def BC_generate(self, inputs):
        X = inputs[:, 0]
        Y = inputs[:, 1]

        u_borders = 4. * tf.math.multiply(self.l - Y, Y) / self.l**2
        Dop = tf.math.sqrt((X - 0.2)**2 + (Y - 0.2)**2)
        theta = tf.math.atan2(Y - 0.2, X - 0.2) + tf.constant(np.pi)
        pi_3 = tf.constant(np.pi / 3.)
        theta_prime = tf.math.floormod(theta + pi_3, 2 * pi_3) - pi_3   
        R = 0.1
        r = 0.05
        y = (- r + np.sqrt(4 * R**2 - 3 * r**2)) / 2.
        gamma = tf.math.sqrt(R**2 - y**2 * tf.math.square(tf.math.sin(theta_prime))) - y * tf.math.cos(theta_prime)
        u_tool = Dop - gamma
        v_par = tf.reduce_prod(tf.stack([X, Y, self.l - Y, u_tool]), axis=0)
        u_par = tf.math.add(tf.math.multiply(u_borders, tf.math.exp(-1000. * X)), v_par)
        p_par = self.L - X

        Duv = v_par
        Dp = p_par

        return tf.stack([u_par, v_par, p_par], axis=-1), tf.stack([Duv, Duv, Dp], axis=-1)


    def call(self, inputs, layer):
        distances, partial_outputs = self.BC_generate(inputs)
        result = tf.math.add(
            partial_outputs,
            tf.math.multiply(layer, distances)
        )
        return result

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
pinn
"""

class PINN(tf.keras.Model):

    @tf.function
    def compute_residuals(self, inputs):

        rho, mu = 1., 2e-2
        u, v, p_x, p_y, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy = inputs

        res_u = tf.math.add_n([
            tf.math.scalar_mul(rho, tf.math.add(tf.math.multiply(u, u_x), tf.math.multiply(v, u_y))),
            p_x,
            tf.math.scalar_mul(tf.math.negative(mu), tf.math.add(u_xx, u_yy))
        ])

        res_v = tf.math.add_n([
            tf.math.scalar_mul(rho, tf.math.add(tf.math.multiply(u, v_x), tf.math.multiply(v, v_y))),
            p_y,
            tf.math.scalar_mul(tf.math.negative(mu), tf.math.add(v_xx, v_yy))
        ])
        res_cons = u_x + v_y

        return res_u, res_v, res_cons

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
                    generated_outputs = self(v, training=True) #neural network generate the predicted values
                    u, v, p = tf.unstack(generated_outputs, axis=1)
                u_x = tape2.gradient(u,x) #calculation of the first spacial derivatives
                u_y = tape2.gradient(u,y)
                v_x = tape2.gradient(v,x)
                v_y = tape2.gradient(v,y)
                p_x = tape2.gradient(p,x)
                p_y = tape2.gradient(p,y)
            u_xx = tape1.gradient(u_x,x) #calculation of the second spacial derivatives
            u_yy = tape1.gradient(u_y,y)
            v_xx = tape1.gradient(v_x,x)
            v_yy = tape1.gradient(v_y,y)

            res_u, res_v, res_cons = self.compute_residuals((u, v, p_x, p_y, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy)) # equation residuals
            loss_u = tf.math.reduce_mean(tf.math.square(res_u)) #taking the mse of the residuals
            loss_v = tf.math.reduce_mean(tf.math.square(res_v))
            loss_cons = tf.math.reduce_mean(tf.math.square(res_cons))

            loss = loss_u + loss_v + loss_cons #summing the mse
        grads = gradtape.gradient(loss, self.trainable_variables) # calculating gradient
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # applying gradient

        return {"loss":loss, "loss_u":loss_u, "loss_v":loss_v, "loss_cons":loss_cons}

"""
Neural Network
"""

inp = tf.keras.Input((2,))
hid = tf.keras.layers.Dense(30, activation="relu")(inp)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(30, activation="relu")(hid)
hid = tf.keras.layers.Dense(3, activation="linear")(hid)
outp = BC_force_layer(L, l)(inp, hid)
#4 hidden layers of 30 neurons activated by ReLU function


"""
main
"""

model = PINN(inp, outp)
opt = tf.keras.optimizers.Adam(learning_rate=2e-3)
model.compile(optimizer=opt)

# model.run_eagerly = True
model.fit(vect, vals, epochs=20, batch_size=1024)
hist = model.history.history # logs metrics

"""
plot
"""

gen_np = model.predict(vect)
u, v, p = gen_np[:, 0], gen_np[:, 1], gen_np[:, 2]

x_on = np.linspace(0., L, 500)
y_on = np.linspace(0., l, 200)
x_on, y_on = np.meshgrid(x_on, y_on)

R = 0.1
r = 0.05
d = 0.2
y = (- r + np.sqrt(4 * R**2 - 3 * r**2)) / 2.
rotmat = np.array([[np.cos(2./3. * np.pi), np.sin(2./3. * np.pi)], [- np.sin(2./3. * np.pi), np.cos(2./3. * np.pi)]])
O1 = np.array([y, 0.])
O2 = np.matmul(rotmat, O1)
O3 = np.matmul(rotmat, O2)
O1, O2, O3 = O1 + np.array([d, d]), O2 + np.array([d, d]), O3 + np.array([d, d])
m1 = ((x_on - O1[0])**2 + (y_on - O1[1])**2) <= (R**2)
m2 = ((x_on - O2[0])**2 + (y_on - O2[1])**2) <= (R**2)
m3 = ((x_on - O3[0])**2 + (y_on - O3[1])**2) <= (R**2)
mask_tool = m1 & m2 & m3

x_off = vect[:, 0]
y_off = vect[:, 1]        


u_on = griddata((x_off, y_off), u, (x_on, y_on), method='linear')
v_on = griddata((x_off, y_off), v, (x_on, y_on), method='linear')
p_on = griddata((x_off, y_off), p, (x_on, y_on), method='linear')
u_on[mask_tool] = np.nan
v_on[mask_tool] = np.nan
p_on[mask_tool] = np.nan

fig1, axs1 = plt.subplots(3)

c1 = axs1[0].imshow(u_on)
fig1.colorbar(c1, ax=axs1[0])
axs1[0].set_title("U (m/s)")
c2 = axs1[1].imshow(v_on)
fig1.colorbar(c2, ax=axs1[1])
axs1[1].set_title("V (m/s)")
c3 = axs1[2].imshow(p_on)
fig1.colorbar(c3, ax=axs1[2])
axs1[2].set_title("P (Pa)")
fig1.suptitle("Learning Result")
fig1.set_figheight(7.4)


plt.show()

for k in hist.keys(): # plotting all metrics
    met = hist[k]
    plt.plot(met, label=k)
plt.legend()
plt.show()