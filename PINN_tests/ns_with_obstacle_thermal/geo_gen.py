from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt

def geo_gen_2d():
    X_v, Y_v = 1.1, 0.4
    x, y, r = 0.2, 0.2, 0.05
    X_s, Y_s, C_s = 100, 25, 50
    X, Y = np.linspace(0., X_v, X_s), np.linspace(0., Y_v, Y_s)
    geo = np.reshape(np.stack([*np.meshgrid(X, Y)], -1), (-1, 2))
    geo = geo[(geo[:, 0] - x)**2 + (geo[:, 1] - y)**2 > r**2, :]
    angles = np.linspace(0., 2 * np.pi, C_s, endpoint=False)
    circle = np.stack([x + r * np.cos(angles), y + r * np.sin(angles)], -1)

    normals_geo = np.full((geo.shape[0]), np.nan)
    normals_geo[geo[:, 1] == 0.] = np.pi / 2.
    normals_geo[geo[:, 1] == Y_v] = 3. * np.pi / 2.
    normals_circle = angles
    full_geo = np.concatenate((geo, circle), 0)
    normals = np.concatenate((normals_geo, normals_circle), 0)



    msk = np.logical_not(np.isnan(normals))
    normals_ids = np.where(msk)

    x_lim = np.stack([full_geo[:, 0] == 0., full_geo[:, 0] == X_v], -1).any(1)
    BC_points_ids = np.where(np.stack([msk, x_lim], -1).any(1))

    return full_geo, normals, BC_points_ids, normals_ids

def geo_gen():
    t_v = 1000.
    t_s = 200
    T = np.linspace(0., t_v, t_s)

    g_2d, n_2d, BC_2d, nid_2d = geo_gen_2d()
    nb = g_2d.shape[0]

    g_3d = np.empty((nb * t_s, 3))
    g_3d[:, 0:2] = np.tile(g_2d, (t_s, 1))
    g_3d[:, 2] = np.tile(T, nb)

    n_3d = np.tile(n_2d, t_s)

    BC_2d_mask = np.full_like(n_2d, False)
    BC_2d_mask[BC_2d] = True
    BC_3d = np.where(np.tile(BC_2d_mask, t_s))

    nid_2d_mask = np.full_like(n_2d, False)
    nid_2d_mask[nid_2d] = True
    nid_3d = np.where(np.tile(nid_2d_mask, t_s))   

    return g_3d, n_3d, BC_3d, *nid_3d



if __name__ == "__main__":

    full_geo, normals, BC_points_ids, msk = geo_gen_2d()

    # fig, ax = plt.subplots(1)
    # ax.scatter(full_geo[:, 0], full_geo[:, 1])
    # ax.axis('equal')
    # plt.show()

    # fig, ax = plt.subplots(1)
    # ax.quiver(full_geo[msk, 0], full_geo[msk, 1], np.cos(normals[msk]), np.sin(normals[msk]))
    # ax.axis('equal')
    # plt.show()

    # fig, ax = plt.subplots(1)
    # ax.scatter(full_geo[BC_points_ids, 0], full_geo[BC_points_ids, 1])
    # ax.axis('equal')
    # plt.show()

    full_geo, normals, BC_points_ids, msk = geo_gen()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
    # ax.scatter(full_geo[:, 0], full_geo[:, 1], full_geo[:, 2])
    # ax.set_xlim3d(0., 1.1)
    # ax.set_ylim3d(0., 1.1)
    # ax.set_zlim3d(0., 1000.)
    # ax.view_init(90, 45)
    # plt.show()

    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
    # xq, yq, zq = full_geo[msk, 0], full_geo[msk, 1], full_geo[msk, 2]
    # uq, vq, wq = np.cos(normals[msk]), np.sin(normals[msk]), np.zeros_like(normals[msk])
    # ax.quiver3D(xq, yq, zq, uq, vq, wq, length=0.05)
    # ax.set_xlim3d(0., 1.1)
    # ax.set_ylim3d(0., 1.1)
    # ax.set_zlim3d(0., 1000.)
    # ax.view_init(90, 45)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
    # ax.scatter(full_geo[BC_points_ids, 0], full_geo[BC_points_ids, 1], full_geo[BC_points_ids, 2])
    # ax.set_xlim3d(0., 1.1)
    # ax.set_ylim3d(0., 1.1)
    # ax.set_zlim3d(0., 1000.)
    # ax.view_init(90, 45)
    # plt.show()