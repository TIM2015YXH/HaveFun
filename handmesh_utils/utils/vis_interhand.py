# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils.transforms import cam2pixel
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
try:
    matplotlib.use('tkagg')
except:
    pass


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:, :2, :2]
        shift = transforms[:, :2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def perspective_np(points, calibrations):
    if points.shape[1] == 2:
        points = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points[:, :3] /= points[:, 2:3]
    points1 = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points_img = np.dot(calibrations, points1.T).T
    points_img = points_img[:, [0, 1, 3]]

    return points_img


def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb4'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index4'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle4'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring4'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky4'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        elif joint_name.startswith('l'):
            rgb_dict[joint_name] = (0, 0, 255)
        elif joint_name.startswith('r'):
            rgb_dict[joint_name] = (255, 0, 0)
        else:
            rgb_dict[joint_name] = (230, 230, 0)

    return rgb_dict


def vis_keypoints(img, kps, score, skeleton, score_thr=0.4, line_width=3, circle_rad = 3, filename=None):

    rgb_dict = get_keypoint_rgb(skeleton)
    if img.shape[-1] != 3:
        img = img.transpose(1,2,0)
    _img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])

    if filename is None:
        return np.array(_img)
    else:
        _img.save(osp.join(filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename=None, score_thr=0.4, line_width=3, circle_rad=3, ax=None, shape=(256, 256), lim=0):

    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(float(shape[1]) / fig.dpi, float(shape[0]) / fig.dpi, forward=True)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name])/255., linewidth=line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c=np.array(rgb_dict[joint_name]).reshape(1, 3)/255., marker='o', s=circle_rad)
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3)/255., marker='o', s=circle_rad)
    ax.scatter([0,], [0,], [0,], c='g', s=circle_rad, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    if lim>0:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    if fig is not None:
        if filename is None:
            ret = fig2data(fig)
            plt.close(fig)
            return ret
        else:
            fig.savefig(filename, dpi=fig.dpi)
            plt.close(fig)


def vis_3d_mesh(verts_list, face_list, hand_type, filename=None, ax=None, shape=(256, 256)):

    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(float(shape[1]) / fig.dpi, float(shape[0]) / fig.dpi, forward=True)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
    verts_valid = []
    if hand_type[0] == 1:
        mesh = Poly3DCollection(verts_list[0][face_list[0]], alpha=0.2)
        face_color = (255 / 255, 0 / 255, 0 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        verts_valid.append(verts_list[0])
    if hand_type[1] == 1:
        mesh = Poly3DCollection(verts_list[1][face_list[1]], alpha=0.2)
        face_color = (0 / 255, 0 / 255, 255 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        verts_valid.append(verts_list[1])
    ax.scatter([0,], [0,], [0,], c='g', s=50, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cam_equal_aspect_3d(ax, np.concatenate(verts_valid, axis=0))
    if fig is not None:
        if filename is None:
            ret = fig2data(fig)
            plt.close(fig)
            return ret
        else:
            fig.savefig(filename, dpi=fig.dpi)
            plt.close(fig)


def vis_3d_sdf(samples, labels, filename=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
    c = np.array(['r'] * samples.shape[0])
    c[labels==0] = 'g'
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=c, s=1, marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if fig is not None:
        if filename is None:
            ret = fig2data(fig)
            return ret
        else:
            fig.savefig(filename, dpi=fig.dpi)


def vis_aligned_mesh(verts_list, face_list, img, focal, princpt, hand_type, ax=None):
    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(float(img.shape[1]) / fig.dpi, float(img.shape[0]) / fig.dpi, forward=True)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = None
    ax.imshow(img)
    ax.axis('off')
    if hand_type[0] == 1:
        verts2uv = cam2pixel(verts_list[0], focal, princpt).astype(np.int32)
        ax.triplot(verts2uv[:, 0], verts2uv[:, 1], face_list[0], lw=0.5, c='r')
    if hand_type[1] == 1:
        verts2uv = cam2pixel(verts_list[1], focal, princpt).astype(np.int32)
        ax.triplot(verts2uv[:, 0], verts2uv[:, 1], face_list[1], lw=0.5, c='b')

    # ax.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)
    if fig is not None:
        plt.show()


def vis_aligned_mesh2d(verts, faces, img, hand_type, ax=None):
    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(float(img.shape[1]) / fig.dpi, float(img.shape[0]) / fig.dpi, forward=True)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = None
    ax.imshow(img)
    ax.axis('off')
    if hand_type[0] == 1:
        ax.triplot(verts[:778, 0], verts[:778, 1], faces[:1538], lw=0.5, c='r')
    if hand_type[1] == 1:
        ax.triplot(verts[778:, 0], verts[778:, 1], faces[1538:], lw=0.5, c='b')

    # ax.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)
    if fig is not None:
        ret = fig2data(fig)
        plt.close()
        return ret

def vis_aligned_sdf(samples, labels, img, calibrations, transforms=None, filename=None, ax=None):
    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(float(img.shape[1]) / fig.dpi, float(img.shape[0]) / fig.dpi, forward=True)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = None
    ax.imshow(img)
    ax.axis('off')
    c = np.array(['r'] * samples.shape[0])
    c[labels==0] = 'g'
    samples = torch.from_numpy(samples).T.float()
    if samples.ndim == 2:
        samples.unsqueeze_(0)
    if calibrations.ndim == 2:
        calibrations.unsqueeze_(0)
    if transforms is not None and transforms.ndim == 2:
        transforms.unsqueeze_(0)
    proj = perspective(samples, calibrations, transforms)[0].numpy().T
    # proj = cam2pixel(samples, focal, princpt).astype(np.int32)
    if transforms is not None:
        proj = (proj + 1) / 2 * img.shape[0]
    proj = proj.astype(np.int32)
    ax.scatter(proj[:, 0], proj[:, 1], c=c, s=1, marker='o')

    if fig is not None:
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


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
