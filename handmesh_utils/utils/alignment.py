import numpy as np
import cv2
from data.FreiHAND.kinematics import mano_to_mpii
from scipy.optimize import minimize


def align_2D3D(vertex, uv, j_regressor, K, size, uv_conf=None, poly=None, xyz_pred=None):
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0., 2)) if vertex.shape[0]<1000 else ((None, None), (None, None), (1, 8))
    poly_protect = [0.06, 0.02] if vertex.shape[0]<1000 else [1.0, 0.5]

    if vertex.shape[0] in [21, 17] or uv.shape[0] in [778, 6980]:
        vertex2xyz = vertex
        try_poly = False
    else:
        if xyz_pred is None:
            vertex2xyz = np.matmul(j_regressor, (vertex, vertex[:778])[vertex.shape[0] in [810, 826]])
            if vertex2xyz.shape[0] == 21:
                vertex2xyz = mano_to_mpii(vertex2xyz)
        else:
            vertex2xyz = xyz_pred
        try_poly = True
    # uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * vertex2xyz[:, 2:]
    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        success = False
    else:
        loss = np.array([5, ])
        attempt = 5
        while loss.mean() > 2 and attempt:
            attempt -= 1
            uv = uv[uv_select.repeat(2, axis=1)].reshape(-1, 2)
            uv_conf = uv_conf[uv_select].reshape(-1, 1)
            vertex2xyz = vertex2xyz[uv_select.repeat(3, axis=1)].reshape(-1, 3)
            sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, K))
            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = np.matmul(K, xyz.T).T
            uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
            loss = abs((proj - uvz).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]

    if poly is not None and try_poly:
        if isinstance(size, list):
            poly = find_1Dproj(poly[0])
            poly[:, :2] /= size[0]
            poly[:, 2:] /= size[1]
        else:
            poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, K, size))
        if sol.success:
            t2 = sol.x
            d = distance(t, t2)
            if d > poly_protect[0]:
                t = t2
            elif d > poly_protect[1]:
                t = t * (1 - (d - poly_protect[1]) / (poly_protect[0] - poly_protect[1])) + t2 * ((d - poly_protect[1]) / (poly_protect[0] - poly_protect[1]))
    if xyz_pred is not None:
        return vertex + t, xyz_pred + t, success
    return vertex + t, success


def align_2D3D_wrist(vertex, K, size, poly):
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0., 2))

    if poly is not None:
        center2 = (vertex+t)[-16:].mean(axis=0, keepdims=True)
        center1 = (vertex+t)[:16].mean(axis=0, keepdims=True)
        c2d = np.matmul(K, np.concatenate([center1, center2], 0).T).T
        c2d = (c2d / c2d[:, 2:])[:, :2]
        c2d_norm = np.array([c2d[1, 1]-c2d[0, 1], c2d[0, 0]-c2d[1, 0]])
        c2d_norm = c2d_norm / np.linalg.norm(c2d_norm)
        c2d_para = np.array([c2d[1, 0]-c2d[0, 0], c2d[1, 1]-c2d[0, 1]])
        c2d_para = c2d_para / np.linalg.norm(c2d_para)

        poly_points = poly[0]
        poly_proj_norm = (poly_points * c2d_norm).sum(axis=1)
        poly_proj_para = (poly_points * c2d_para).sum(axis=1)
        poly_proj = np.array([poly_proj_norm.min(), poly_proj_norm.max(), poly_proj_para.min()])
        # poly_proj = poly_proj / np.array(size)

        sol = minimize(align_wrist, t, method='SLSQP', bounds=bounds, args=(poly_proj, vertex, K, c2d_norm, c2d_para, size))
        if sol.success:
            t = sol.x
            center1_post = (vertex+t)[:16].mean(axis=0)
            center1_post_norm = center1_post / np.linalg.norm(center1_post)
            t = t + 0.1 * center1_post_norm

        return vertex + t, c2d_norm, c2d_para

    return vertex + t, None, None


def distance(x, y):
    return np.sqrt(((x - y)**2).sum())


def find_box(points):
    # x = sorted(points[:, 0])
    # y = sorted(points[:, 1])
    # return np.array([x[0], y[0], x[-1], y[-1]])
    return np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])


def find_1Dproj(points):
    angles = [(0, 90), (-15, 75), (-30, 60), (-45, 45), (-60, 30), (-75, 15)]
    axs = [(np.array([[np.cos(x/180*np.pi), np.sin(x/180*np.pi)]]), np.array([np.cos(y/180*np.pi), np.sin(y/180*np.pi)])) for x, y in angles]
    proj = []
    for ax in axs:
        x = (points * ax[0]).sum(axis=1)
        y = (points * ax[1]).sum(axis=1)
        proj.append([x.min(), x.max(), y.min(), y.max()])

    return np.array(proj)


def align_box(t, box, vertex, K, size):
    proj = np.matmul(K, (vertex + t).T).T
    proj = (proj / proj[:, 2:])[:, :2]
    if isinstance(size, list):
        proj_box = find_1Dproj(proj)
        proj_box[:, :2] /= size[0]
        proj_box[:, 2:] /= size[1]
    else:
        proj_box = find_box(proj) / size
    loss = (proj_box - box)**2

    return loss.mean()

def align_wrist(t, poly_proj, vertex, K, ax_norm, ax_para, size):
    proj2d = np.matmul(K, (vertex+ t).T).T
    proj2d = (proj2d / proj2d[:, 2:])[:, :2]

    proj_norm = (proj2d * ax_norm).sum(axis=1)
    proj_para = (proj2d * ax_para).sum(axis=1)
    proj = np.array([proj_norm.min(), proj_norm.max(), proj_para.min()])
    # proj /= np.array(size)

    loss = (proj - poly_proj)**2

    return loss.mean()

def align_poly(t, poly, vertex, K, size):
    proj = np.matmul(K, (vertex + t).T).T
    proj = (proj / proj[:, 2:])[:, :2]
    if isinstance(size, list):
        proj = find_1Dproj(proj)
        proj[:, :2] /= size[0]
        proj[:, 2:] /= size[1]
    else:
        proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2

    return loss.mean()


def align_uv(t, uv, vertex2xyz, K):
    xyz = vertex2xyz + t
    proj = np.matmul(K, xyz.T).T
    uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
    loss = (proj - uvz)**2

    return loss.mean()


def map2uv(map, size=(224, 224)):
    if map.ndim == 4:
        uv = np.zeros((map.shape[0], map.shape[1], 2))
        uv_conf = np.zeros((map.shape[0], map.shape[1], 1))
        map_size = map.shape[2:]
        for j in range(map.shape[0]):
            for i in range(map.shape[1]):
                uv_conf[j][i] = map[j, i].max()
                max_pos = map[j, i].argmax()
                uv[j][i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
                uv[j][i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]
    else:
        uv = np.zeros((map.shape[0], 2))
        uv_conf = np.zeros((map.shape[0], 1))
        map_size = map.shape[1:]
        for i in range(map.shape[0]):
            uv_conf[i] = map[i].max()
            max_pos = map[i].argmax()
            uv[i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
            uv[i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]

    return uv, uv_conf


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x