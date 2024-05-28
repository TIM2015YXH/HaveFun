from __future__ import unicode_literals, print_function
import transforms3d as t3d
import numpy as np
import cv2
try:
    from opendr.lighting import LambertianPointLight
    from opendr.camera import ProjectPoints
    from utils.render import ColoredRenderer
except:
    print('Fail to import opendr render')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from data.FreiHAND.kinematics import mano_to_mpii
from scipy.optimize import minimize
from utils.fh_utils import plot_hand
from utils.draw3d import save_a_image_with_mesh_joints
import os
import matplotlib
matplotlib.use('TKAgg', force=False)


def get_focal_pp(K):
    """ Extract the camera parameters that are relevant for an orthographic assumption. """
    focal = 0.5 * (K[0, 0] + K[1, 1])
    pp = K[:2, 2]
    return focal, pp


def backproject_ortho(uv, scale,  # kind of the predictions
                      focal, pp):  # kind of the camera calibration
    """ Calculate 3D coordinates from 2D coordinates and the camera parameters. """
    uv = uv.copy()
    uv -= pp
    xyz = np.concatenate([np.reshape(uv, [-1, 2]),
                          np.ones_like(uv[:, :1])*focal], 1)
    xyz /= scale
    return xyz


def recover_root(uv_root, scale,
                 focal, pp):
    uv_root = np.reshape(uv_root, [1, 2])
    xyz_root = backproject_ortho(uv_root, scale, focal, pp)
    return xyz_root


def render(cam_intrinsics, V, F,  dist=None, M=None, img_shape=None, render_mask=False):

    if dist is None:
        dist = np.zeros(5)
    dist = dist.flatten()
    if M is None:
        M = np.eye(4)

    # get R, t from M (has to be world2cam)
    R = M[:3, :3]
    ax, angle = t3d.axangles.mat2axangle(R)
    rt = ax*angle
    rt = rt.flatten()
    t = M[:3, 3]

    w, h = (224, 224)
    if img_shape is not None:
        w, h = img_shape[1], img_shape[0]

    pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])
    f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])

    # Create OpenDR renderer
    rn = ColoredRenderer()

    # Assign attributes to renderer
    rn.camera = ProjectPoints(rt=rt,
                              t=t, # camera translation
                              f=f,  # focal lengths
                              c=pp,  # camera center (principal point)
                              k=dist)  # OpenCv distortion params
    rn.frustum = {'near': 0.1, 'far': 5., 'width': w, 'height': h}

    # V, F = self._get_verts_faces()
    rn.set(v=V,
           f=F,
           bgcolor=np.zeros(3))

    if render_mask:
        rn.vc = np.ones_like(V)  #for segmentation mask like rendering
    else:
        colors = np.ones_like(V)

        # Construct point light sources
        rn.vc = LambertianPointLight(f=F,
                                     v=V,
                                     num_verts=V.shape[0],
                                     light_pos=np.array([-1000, -1000, -2000]),
                                     vc=0.8 * colors,
                                     light_color=np.array([1., 1., 1.]))

        rn.vc += LambertianPointLight(f=F,
                                      v=V,
                                      num_verts=V.shape[0],
                                      light_pos=np.array([1000, 1000, -2000]),
                                      vc=0.25 * colors,
                                      light_color=np.array([1., 1., 1.]))

        rn.vc += LambertianPointLight(f=F,
                                      v=V,
                                      num_verts=V.shape[0],
                                      light_pos=np.array([2000, 2000, 2000]),
                                      vc=0.1 * colors,
                                      light_color=np.array([1., 1., 1.]))

        rn.vc += LambertianPointLight(f=F,
                                      v=V,
                                      num_verts=V.shape[0],
                                      light_pos=np.array([-2000, -2000, 2000]),
                                      vc=0.1 * colors,
                                      light_color=np.array([1., 1., 1.]))

    # print(rn.r.max())
    # render
    img = (np.array(rn.r) * 255).astype(np.uint8)
    return img


def tensor2array(tensor, max_value=None, colormap='jet', channel_first=True, mean=0.5, std=0.5):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'jet':
                colormap = cv2.COLORMAP_JET
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = ((mean + tensor.numpy() * std) * 255).astype(np.uint8)
        if not channel_first:
            array = array.transpose(1, 2, 0)

    return array


def base_transform(img, size, mean=0.5, std=0.5):
    if not isinstance(size, tuple):
        size = (size, size)
    x = cv2.resize(img, size).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x


def inv_based_tranmsform(x, mean=0.5, std=0.5):
    x = x.transpose(1, 2, 0)
    image = (x * std + mean) * 255
    return image.astype(np.uint8)


def img_vertex_align(img, vertex, K, poly=None, face=None, uv=None, vis=False, save_path=None, step=0, vertex2xyz=None, renderer=None, mask=None):
    vertex2uv = np.matmul(K, vertex.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(np.int)
    img_mask = img.copy()
    if mask is not None:
        mask = np.concatenate([mask[:, :, None], np.zeros(list(mask.shape)+[2])], 2).astype(np.uint8) * 255
        img_mask = cv2.addWeighted(img_mask, 1, mask, 0.5, 0) # np.ones_like(img_mask)*255
    if poly is not None:
        cv2.polylines(img_mask, poly, isClosed=True, thickness=2, color=(255, 0, 0)) # (255, 153, 102)
    if vis:
        plt.figure()
        ax = plt.gca()
        plt.imshow(img_mask)
        plt.axis('off')
        if face is None:
            plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], 'o', color='green', markersize=1)
        else:
            plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=0.2, color=(157/255, 187/255, 97/255))
        if uv is not None:
            plot_hand(ax, uv[:, ::-1])
        plt.show()
    if save_path and renderer:
        save_a_image_with_mesh_joints(renderer, img[:, :, ::-1], img_mask[:, :, ::-1], np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
                                      np.array([0, 0, img.shape[1], img.shape[0]]), vertex, uv, vertex2xyz, face, os.path.join(save_path, str(step)+'_plot.jpg'))
        # cv2.imwrite(os.path.join(save_path, str(step) + '_img.jpg'), img_mask[:, :, ::-1])


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

def align_2D3D_mask(vertex, K, size, poly):
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0., 2))

    if poly is not None:
        if isinstance(size, list):
            poly = find_1Dproj(poly[0])
            poly[:, :2] /= size[0]
            poly[:, 2:] /= size[1]
        else:
            poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, t, method='SLSQP', bounds=bounds, args=(poly, vertex, K, size))
        if sol.success:
            t = sol.x

    return vertex + t


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


def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))
            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map


def z2map(z, harf_size=28, clip_value=1):
    z = np.clip(z, -clip_value, clip_value)
    z = (z * harf_size / clip_value + harf_size).astype(np.uint8)
    kernel_size = ((harf_size * 2) * 13 // (harf_size * 2) -1) // 2
    gaussian_map = np.zeros((z.shape[0], (harf_size * 2)))
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)[:, 0]
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        s_pt = np.array(z[i])
        p_start = s_pt - kernel_size
        p_end = s_pt + kernel_size
        p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
        k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
        p_end_fix = (p_end <= ((harf_size * 2) - 1)) * p_end + (p_end > ((harf_size * 2) - 1)) * ((harf_size * 2) - 1)
        k_end_fix = (p_end <= ((harf_size * 2) - 1)) * kernel_size * 2 + (p_end > ((harf_size * 2) - 1)) * (2 * kernel_size - (p_end - ((harf_size * 2) - 1)))
        gaussian_map[i, p_start_fix: p_end_fix + 1] = gaussian_kernel[k_start_fix: k_end_fix + 1]

    return gaussian_map


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


def map2uv_centroid(map, size=(224, 224)):

    def softmax(x):
        valid_idx = x > x.max()*0.3
        valid_val = x[valid_idx]
        exp_val = np.exp(valid_val/valid_val.max())
        softmax_val = exp_val / np.sum(exp_val)
        softmax_x = np.zeros_like(x)
        softmax_x[valid_idx] = softmax_val
        return softmax_x

    if map.ndim == 4:
        uv = np.zeros((map.shape[0], map.shape[1], 2))
        uv_conf = np.zeros((map.shape[0], map.shape[1], 1))
        map_size = map.shape[2:]
        for j in range(map.shape[0]):
            for i in range(map.shape[1]):
                uv_conf[j][i] = map[j, i].max()
                soft_max = softmax(map[j, i])
                pt_u, pt_v = 0, 0
                for m in range(map[j][i].shape[0]):
                    for n in range(map[j][i].shape[1]):
                        pt_u += soft_max[m, n] * n
                        pt_v += soft_max[m, n] * m
                uv[j][i][1] = pt_v / map_size[0] * size[0]
                uv[j][i][0] = pt_u / map_size[1] * size[1]
    else:
        uv = np.zeros((map.shape[0], 2))
        uv_conf = np.zeros((map.shape[0], 1))
        map_size = map.shape[1:]
        for i in range(map.shape[0]):
            uv_conf[i] = map[i].max()
            soft_max = softmax(map[i])
            pt_u, pt_v = 0, 0
            for m in range(map[i].shape[0]):
                for n in range(map[i].shape[1]):
                    pt_u += soft_max[m, n] * n
                    pt_v += soft_max[m, n] * m
            uv[i][1] = pt_v / map_size[0] * size[0]
            uv[i][0] = pt_u / map_size[1] * size[1]

    return uv, uv_conf


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts = hand_info['verts']
    joints = hand_info.get('joints')
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if joints is not None:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    ax.scatter([0, ], [0, ], [0, ], color='g')
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
