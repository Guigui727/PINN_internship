import numpy as np
import matplotlib.pyplot as plt

def tool():
    # R = 0.005
    # l = 0.0025
    # margin = 0.0006
    R = 0.1
    r = 0.05
    margin = 0.006

    y = (- r + np.sqrt(4 * R**2 - 3 * r**2)) / 2.
    pi3 = np.pi / 3.
    theta = np.linspace(pi3, - pi3, 50, endpoint=False)
    cos  = np.cos(theta)
    sin = np.sin(theta)
    gamma = np.sqrt(R**2 - y**2 * np.square(sin)) - y * cos
    X = np.multiply(gamma, cos)
    Y = np.multiply(gamma, sin)

    vect1 = np.stack([X, Y], axis=1)

    rotmat = np.array([[np.cos(2. * pi3), np.sin(2. * pi3)], [- np.sin(2. * pi3), np.cos(2. * pi3)]])

    vect2 = np.matmul(vect1, rotmat)
    vect3 = np.matmul(vect2, rotmat)

    border_pts = 10
    pts1 = np.zeros((gamma.shape[0], border_pts, 2))

    for i in range(gamma.shape[0]):
        borders = np.linspace(r + margin, gamma[i], border_pts, endpoint=False)
        pts1[i, :, 0] = np.multiply(borders, cos[i])
        pts1[i, :, 1] = np.multiply(borders, sin[i])

    pts1 = pts1.reshape((pts1.shape[0] * pts1.shape[1], 2))
    pts2 = np.matmul(pts1, rotmat)
    pts3 = np.matmul(pts2, rotmat)

    return np.concatenate([vect1, vect2, vect3], axis=0), np.concatenate([pts1, pts2, pts3], axis=0)


def body(displacement = 0.):

    H, W = 0.40, 1.1
    n_H, n_W = 128, 387
    l = np.linspace(0., H, n_H)
    L = np.linspace(0., W, n_W)

    L, l = np.meshgrid(L, l)

    pts = np.stack([L, l], axis=-1)
    pts = pts.reshape((pts.shape[0] * pts.shape[1], 2))
    select_non_border = np.logical_not(np.any(np.stack([pts[:, 0] == 0., pts[:, 0] == W, pts[:, 1] == 0., pts[:, 1] == H], axis=-1), axis=1))
    
    pts[select_non_border] = pts[select_non_border] + displacement * np.array([[W / float(n_W), H / float(n_H)]]) * np.random.randn(*pts[select_non_border].shape)

    dist = np.sum(
        np.square(
            np.subtract(pts, np.array([[0.2, 0.2]]))
        ),
        axis=1
    )
    
    pts = pts[dist > (0.05 + 0.008)**2]

    return pts






if __name__ == "__main__":

    t, s = tool() 
    t, s = t + np.array([[0.2, 0.2]]), s + np.array([[0.2, 0.2]])
    b = body(0.1)

    fig, ax = plt.subplots(1)
    ax.scatter(t[:, 0], t[:, 1], color='r', marker='x')
    ax.scatter(s[:, 0], s[:, 1], color='b', marker='x')
    ax.scatter(b[:, 0], b[:, 1], color='g', marker='x')
    ax.axis('equal')
    plt.show()
    

