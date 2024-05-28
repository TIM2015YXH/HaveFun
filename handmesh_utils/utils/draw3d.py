# Copyright (c) Liuhao Ge. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

h36m_joint_line = [[[0, 1], [1, 2], [2, 3]],
                   [[0, 4], [4, 5], [5, 6]],
                   [[0, 7], [7, 8], [8, 9], [9, 10]],
                   [[8, 11], [11, 12], [12, 13]],
                   [[8, 14], [14, 15], [15, 16]]]

color_h36m_joints = [[0.0, 0.5, 0.5],
                     [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],
                     [0.0, 0.6, 0.6], [0.0, 0.7, 0.7], [0.0, 0.8, 0.8], [0.0, 0.9, 0.9],
                     [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0],
                     [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0]]

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


def draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz, vps=(60.0, -60.0), axis='y'):
    """
    :param mesh_renderer:
    :param image: H x W x 3
    :param cam_param: fx, fy, u0, v0
    :param box: x, y, w, h
    :param mesh_xyz: M x 3
    :param vps: rotate angle
    :param axis: rotate axis
    :return:
    """
    resize_ratio = float(image.shape[0]) / box[2]
    cam_for_render = np.array([cam_param[0], cam_param[2] - box[0], cam_param[3] - box[1]]) * resize_ratio

    rend_img_overlay = mesh_renderer(mesh_xyz, cam=cam_for_render, img=image, do_alpha=True)
    if len(vps) == 0:
        return rend_img_overlay

    rend_img_vps = [mesh_renderer.rotated(mesh_xyz, vp, axis=axis, cam=cam_for_render, img_size=image.shape[:2]) for vp in vps]
    return rend_img_overlay, rend_img_vps[0], rend_img_vps[1]

def draw_aligned_mesh_plt(image, cam_param, mesh_xyz, face, lw=0.5):
    """
    :param image: H x W x 3
    :param cam_param: 1 x 3 x 3
    :param mesh_xyz: 778 x 3
    :param face: 1538 x 3 x 2
    :return:
    """
    vertex2uv = np.matmul(cam_param, mesh_xyz.T).T
    vertex2uv = (vertex2uv / vertex2uv[:, 2:3])[:, :2].astype(np.int)

    fig = plt.figure()
    fig.set_size_inches(float(image.shape[1]) / fig.dpi, float(image.shape[0]) / fig.dpi, forward=True)
    plt.imshow(image)
    plt.axis('off')
    if face is None:
        plt.plot(vertex2uv[:, 0], vertex2uv[:, 1], 'o', color='green', markersize=1)
    else:
        plt.triplot(vertex2uv[:, 0], vertex2uv[:, 1], face, lw=lw, color='orange')

    plt.subplots_adjust(left=0., right=1., top=1., bottom=0, wspace=0, hspace=0)
    ret = fig2data(fig)
    plt.close(fig)

    return ret

def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] in [21, 17]
    skeleton_overlay = image.copy()
    marker_sz = 5
    line_wd = 2
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        if pose_uv.shape[0] == 21:
            cv2.circle(
                skeleton_overlay, joint,
                radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
                cv2.line(
                    skeleton_overlay, root_joint, joint,
                    color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
            else:
                joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
                cv2.line(
                    skeleton_overlay, joint_2, joint,
                    color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            cv2.circle(
                skeleton_overlay, joint,
                radius=marker_sz, color=color_h36m_joints[joint_ind] * np.array(255), thickness=-1,
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
            for i, line in enumerate(h36m_joint_line):
                for j, sub_line in enumerate(line):
                    joint0 = pose_uv[sub_line[0], 0].astype('int32'), pose_uv[sub_line[0], 1].astype('int32')
                    joint1 = pose_uv[sub_line[1], 0].astype('int32'), pose_uv[sub_line[1], 1].astype('int32')
                    cv2.line(
                        skeleton_overlay, joint0, joint1,
                        color=color_h36m_joints[sub_line[0]] * np.array(255), thickness=int(line_wd),
                        lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay


# def draw_3d_skeleton(pose_cam_xyz, image_size):
#     """
#     :param pose_cam_xyz: 21 x 3
#     :param image_size: H, W
#     :return:
#     """
#     assert pose_cam_xyz.shape[0] in [21, 17]
#     # pose_cam_xyz = pose_cam_xyz[:, [2, 1, 0]]
#     fig = plt.figure(figsize=(5, 5))
#     fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
#
#     ax = plt.subplot(111, projection='3d')
#     marker_sz = 15
#     line_wd = 2
#
#     for joint_ind in range(pose_cam_xyz.shape[0]):
#         if pose_cam_xyz.shape[0] == 21:
#             ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
#                     pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
#             if joint_ind == 0:
#                 continue
#             elif joint_ind % 4 == 1:
#                 ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
#                         color=color_hand_joints[joint_ind], linewidth=line_wd)
#             else:
#                 ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
#                         pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
#                         linewidth=line_wd)
#         else:
#             ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
#                     pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_h36m_joints[joint_ind], markersize=marker_sz)
#             for i, line in enumerate(h36m_joint_line):
#                 for j, sub_line in enumerate(line):
#                     ax.plot(pose_cam_xyz[[sub_line[0], sub_line[1]], 0], pose_cam_xyz[[sub_line[0], sub_line[1]], 1],
#                             pose_cam_xyz[[sub_line[0], sub_line[1]], 2], color=color_h36m_joints[sub_line[0]], linewidth=line_wd)
#
#     ax.axis('auto')
#     if pose_cam_xyz.shape[0] == 21:
#         lim = 0.1
#         z_lim = [0.3, 0.8, 0.1]
#     else:
#         lim = 1
#         z_lim = [3, 8, 1]
#     ticks = np.arange(-lim, lim, step=lim/5)
#     z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
#     plt.xticks(ticks, [-lim, '', '', '', '', 0, '', '', '', lim], fontsize=14)
#     plt.yticks(ticks, [-lim, '', '', '', '', 0, '', '', '', lim], fontsize=14)
#     ax.set_zticks(z_ticks)
#     ax.set_zticklabels(['']*(z_ticks.shape[0]), fontsize=14)
#     # plt.grid(True)
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('Z')
#     ax.view_init(elev=-65, azim=-65)
#     plt.subplots_adjust(left=0., right=0.95, top=0.9, bottom=0, wspace=0, hspace=0)
#     # plt.show()
#
#     ret = fig2data(fig)  # H x W x 4
#     plt.close(fig)
#     return ret


def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: W, H
    :return:
    """
    assert pose_cam_xyz.shape[0] in [21, 17]
    # pose_cam_xyz = pose_cam_xyz[:, [2, 1, 0]]
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 5
    line_wd = 1

    ax.plot([-0.05, 0.05, 0.05, -0.05, -0.05], [-0.05, -0.05, 0.05, 0.05, -0.05], [0, 0, 0, 0, 0],
            color=(0 / 255, 0 / 255, 200 / 255))
    ax.plot([0.05, 0], [0.05, 0], [0, -0.1], color=(0 / 255, 0 / 255, 200 / 255), linestyle=':')
    ax.plot([0.05, 0], [-0.05, 0], [0, -0.1], color=(0 / 255, 0 / 255, 200 / 255), linestyle=':')
    ax.plot([-0.05, 0], [-0.05, 0], [0, -0.1], color=(0 / 255, 0 / 255, 200 / 255), linestyle=':')
    ax.plot([-0.05, 0], [0.05, 0], [0, -0.1], color=(0 / 255, 0 / 255, 200 / 255), linestyle=':')

    for joint_ind in range(pose_cam_xyz.shape[0]):
        if pose_cam_xyz.shape[0] == 21:
            ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                    pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                        color=color_hand_joints[joint_ind], linewidth=line_wd)
            else:
                ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                        pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                        linewidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                    pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_h36m_joints[joint_ind], markersize=marker_sz)
            for i, line in enumerate(h36m_joint_line):
                for j, sub_line in enumerate(line):
                    ax.plot(pose_cam_xyz[[sub_line[0], sub_line[1]], 0], pose_cam_xyz[[sub_line[0], sub_line[1]], 1],
                            pose_cam_xyz[[sub_line[0], sub_line[1]], 2], color=color_h36m_joints[sub_line[0]], linewidth=line_wd)

    fontsize = 10
    ax.axis('auto')
    if pose_cam_xyz.shape[0] == 21:
        lim_x = 0.5
        step_x = lim_x / 15
        x_lim = [-lim_x, lim_x + 0.01, step_x]
        lim_y = 0.5
        step_y = lim_y / 15
        y_lim = [-lim_y, lim_y + 0.01, step_y]
        lim_z = 2.0
        z_lim = [0.0, lim_z + 0.01, 0.2]
        x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
        y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
        z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
        # x_label = [-lim_x, '', '', '', '', 0, '', '', '', lim_x, '']
        # y_label = [-lim_y, '', '', '', '', 0, '', '', '', lim_y, '']
        x_label = [''] * (x_ticks.shape[0])
        y_label = [''] * (y_ticks.shape[0])
        plt.xticks(x_ticks, x_label, fontsize=fontsize)
        plt.yticks(y_ticks, y_label, fontsize=fontsize)
        ax.set_zticks(z_ticks)
        z_ticks = [''] * (z_ticks.shape[0])
        # z_ticks[4] = 0.4
        ax.set_zticklabels(z_ticks, fontsize=fontsize)
    else:
        lim = 1.
        step = lim / 5
        x_lim = [-lim, lim + 0.01, step]
        y_lim = [-lim, lim + 0.01, step]
        lim_z = 5
        z_lim = [0, lim_z+0.1, lim_z/10]
        x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
        y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
        z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
        plt.xticks(x_ticks, [int(-lim), '', '', '', '', 0, '', '', '', '', int(lim)], fontsize=fontsize)
        plt.yticks(y_ticks, [int(-lim), '', '', '', '', 0, '', '', '', '', int(lim)], fontsize=fontsize)
        ax.set_zticks(z_ticks)
        z_label = [''] * (z_ticks.shape[0])
        z_label[-5] = 15
        ax.set_zticklabels(z_label, fontsize=fontsize)
    ax.view_init(elev=140, azim=80)
    plt.subplots_adjust(left=-0.06, right=0.98, top=0.93, bottom=-0.07, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)
    return ret

def draw_3d_mesh(mesh_xyz, image_size, face):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: W, H
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')

    triang = mtri.Triangulation(mesh_xyz[:, 0], mesh_xyz[:, 1], triangles=face)
    ax.plot_trisurf(triang, mesh_xyz[:, 2], color=(145/255, 181/255, 255/255))

    # ax.plot([0.05, 0.05, -0.05, -0.05], [0.05, -0.05, -0.05, 0.05], [0, 0, 0, 0], color=(192/255, 112/255, 0/255))
    ax.plot([-0.05, 0.05, 0.05, -0.05, -0.05], [-0.05, -0.05, 0.05, 0.05, -0.05], [0, 0, 0, 0, 0], color=(0/255, 0/255, 200/255))
    ax.plot([0.05, 0], [0.05, 0], [0, -0.1], color=(0/255, 0/255, 200/255), linestyle=':')
    ax.plot([0.05, 0], [-0.05, 0], [0, -0.1], color=(0/255, 0/255, 200/255), linestyle=':')
    ax.plot([-0.05, 0], [-0.05, 0], [0, -0.1], color=(0/255, 0/255, 200/255), linestyle=':')
    ax.plot([-0.05, 0], [0.05, 0], [0, -0.1], color=(0/255, 0/255, 200/255), linestyle=':')

    fontsize=10
    ax.axis('auto')
    if mesh_xyz.shape[0] < 1000:
        lim_x = 0.5
        step_x = lim_x / 15
        x_lim = [-lim_x, lim_x + 0.01, step_x]
        lim_y = 0.5
        step_y = lim_y / 15
        y_lim = [-lim_y, lim_y + 0.01, step_y]
        lim_z = 2.0
        z_lim = [0.0, lim_z + 0.01, 0.2]
        x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
        y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
        z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
        # x_label = [-lim_x, '', '', '', '', 0, '', '', '', lim_x, '']
        # y_label = [-lim_y, '', '', '', '', 0, '', '', '', lim_y, '']
        x_label = [''] * (x_ticks.shape[0])
        y_label = [''] * (y_ticks.shape[0])
        plt.xticks(x_ticks, x_label, fontsize=fontsize)
        plt.yticks(y_ticks, y_label, fontsize=fontsize)
        # plt.xticks(x_ticks, [''] * (x_ticks.shape[0]), fontsize=14)
        # plt.yticks(y_ticks, [''] * (y_ticks.shape[0]), fontsize=14)
        ax.set_zticks(z_ticks)
        z_label = [''] * (z_ticks.shape[0])
        # z_label[4] = 0.4
        ax.set_zticklabels(z_label, fontsize=fontsize)
    else:
        lim = 1
        step = lim / 5
        x_lim = [-lim, lim + 0.01, step]
        y_lim = [-lim, lim + 0.01, step]
        lim_z = 5
        z_lim = [0, lim_z+0.1, lim_z/10]
        x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
        y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
        z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
        plt.xticks(x_ticks, [int(-lim), '', '', '', '', 0, '', '', '', '', int(lim)], fontsize=fontsize)
        plt.yticks(y_ticks, [int(-lim), '', '', '', '', 0, '', '', '', '', int(lim)], fontsize=fontsize)
        ax.set_zticks(z_ticks)
        z_ticks = [''] * (z_ticks.shape[0])
        z_ticks[-5] = 15
        ax.set_zticklabels(z_ticks, fontsize=fontsize)

    ax.view_init(elev=140, azim=80)
    plt.subplots_adjust(left=-0.06, right=0.98, top=0.93, bottom=-0.07, wspace=0, hspace=0)

    ret = fig2data(fig)
    plt.close(fig)
    return ret

def draw_rel_mesh(mesh_xyz, image_size, face):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')

    triang = mtri.Triangulation(mesh_xyz[:, 0], mesh_xyz[:, 1], triangles=face)
    ax.plot_trisurf(triang, mesh_xyz[:, 2], color=(145/255, 181/255, 255/255))

    lim = 0.15
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.view_init(elev=70, azim=80)
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ret = fig2data(fig)
    plt.close(fig)
    return ret

def draw_rel_pose(pose_rel_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_rel_xyz.shape[0] in [21, 17]
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 10
    line_wd = 2


    for joint_ind in range(pose_rel_xyz.shape[0]):
        if pose_rel_xyz.shape[0] == 21:
            ax.plot(pose_rel_xyz[joint_ind:joint_ind + 1, 0], pose_rel_xyz[joint_ind:joint_ind + 1, 1],
                    pose_rel_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
            if joint_ind == 0:
                continue
            elif joint_ind % 4 == 1:
                ax.plot(pose_rel_xyz[[0, joint_ind], 0], pose_rel_xyz[[0, joint_ind], 1], pose_rel_xyz[[0, joint_ind], 2],
                        color=color_hand_joints[joint_ind], linewidth=line_wd)
            else:
                ax.plot(pose_rel_xyz[[joint_ind - 1, joint_ind], 0], pose_rel_xyz[[joint_ind - 1, joint_ind], 1],
                        pose_rel_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                        linewidth=line_wd)
        else:
            ax.plot(pose_rel_xyz[joint_ind:joint_ind + 1, 0], pose_rel_xyz[joint_ind:joint_ind + 1, 1],
                    pose_rel_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_h36m_joints[joint_ind], markersize=marker_sz)
            for i, line in enumerate(h36m_joint_line):
                for j, sub_line in enumerate(line):
                    ax.plot(pose_rel_xyz[[sub_line[0], sub_line[1]], 0], pose_rel_xyz[[sub_line[0], sub_line[1]], 1],
                            pose_rel_xyz[[sub_line[0], sub_line[1]], 2], color=color_h36m_joints[sub_line[0]], linewidth=line_wd)

    lim = 0.15
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.view_init(elev=70, azim=80)
    plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ret = fig2data(fig)
    plt.close(fig)
    return ret

def save_batch_image_with_mesh_joints(mesh_renderer, batch_images, cam_params, bboxes,
                                      est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz,
                                      file_name, padding=2):
    """
    :param mesh_renderer:
    :param batch_images: B x H x W x 3 (torch.Tensor)
    :param cam_params: B x 4 (torch.Tensor)
    :param bboxes: B x 4 (torch.Tensor)
    :param est_mesh_cam_xyz: B x 1280 x 3 (torch.Tensor)
    :param est_pose_uv: B x 21 x 2 (torch.Tensor)
    :param est_pose_cam_xyz: B x 21 x 3 (torch.Tensor)
    :param file_name:
    :param padding:
    :return:
    """
    num_images = batch_images.shape[0]
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    num_column = 6

    grid_image = np.zeros((num_images * (image_height + padding), num_column * (image_width + padding), 3),
                          dtype=np.uint8)

    for id_image in range(num_images):
        image = batch_images[id_image].numpy()
        cam_param = cam_params[id_image].numpy()
        box = bboxes[id_image].numpy()
        mesh_xyz = est_mesh_cam_xyz[id_image].numpy()
        pose_uv = est_pose_uv[id_image].numpy()
        pose_xyz = est_pose_cam_xyz[id_image].numpy()

        rend_img_overlay, rend_img_vp1, rend_img_vp2 = draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz)
        skeleton_overlay = draw_2d_skeleton(image, pose_uv)
        skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])

        img_list = [image, rend_img_overlay, rend_img_vp1, rend_img_vp2, skeleton_overlay, skeleton_3d]

        height_begin = (image_height + padding) * id_image
        height_end = height_begin + image_height
        width_begin = 0
        width_end = image_width
        for show_img in img_list:
            grid_image[height_begin:height_end, width_begin:width_end, :] = show_img[..., :3]
            width_begin += (image_width + padding)
            width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)


def save_a_image_with_mesh_joints(mesh_renderer, image, img_mask, cam_param, box, mesh_xyz, pose_uv, pose_xyz, face, file_name, pose_uv_prior=None, ret=False):
    """
    :param mesh_renderer:
    :param batch_images: B x H x W x 3 (torch.Tensor)
    :param cam_params: B x 4 (torch.Tensor)
    :param bboxes: B x 4 (torch.Tensor)
    :param est_mesh_cam_xyz: B x 1280 x 3 (torch.Tensor)
    :param est_pose_uv: B x 21 x 2 (torch.Tensor)
    :param est_pose_cam_xyz: B x 21 x 3 (torch.Tensor)
    :param file_name:
    :return:
    """

    rend_img_overlay, rend_img_vp1, rend_img_vp2 = draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz)
    skeleton_overlay = draw_2d_skeleton(image, pose_uv)
    skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])
    mesh_3d = draw_3d_mesh(mesh_xyz, image.shape[:2], face)


    img_list = [img_mask, skeleton_overlay, rend_img_overlay, rend_img_vp1, mesh_3d, skeleton_3d]
    if pose_uv_prior is not None:
        skeleton_overlay_prior = draw_2d_skeleton(image, pose_uv_prior)
        img_list = [skeleton_overlay_prior, ] + img_list
    image_height = image.shape[0]
    image_width = image.shape[1]
    num_column = len(img_list)

    row = 2
    col = num_column // row
    grid =(row, col)
    grid_image = np.zeros((image_height * grid[0], image_width * grid[1], 3), dtype=np.uint8)

    width_begin, width_end = 0, image_width
    height_begin, height_end = 0, image_height
    for i, show_img in enumerate(img_list):
        if i > 0 and i % col == 0:
            height_begin += image_height
            height_end = height_begin + image_height
            width_begin, width_end = 0, image_width
        grid_image[height_begin:height_end, width_begin:width_end, :] = show_img[..., :3]
        width_begin += image_width
        width_end = width_begin + image_width
    if ret:
        return grid_image
    cv2.imwrite(file_name, grid_image)


def draw_2d3d_plt(img, uv, K, verts, face, bbox=None, poly=None, file_name=None):
    plot2d = img.copy()
    plot3d = img.copy()
    if bbox is not None:
        cv2.rectangle(plot2d, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    plot2d = draw_2d_skeleton(plot2d, uv)
    if poly is not None:
        cv2.polylines(plot2d, poly, isClosed=True, thickness=4, color=(255, 255, 0))
    plot3d = draw_aligned_mesh_plt(plot3d, K, verts, face)
    display = np.concatenate([plot2d, plot3d[..., :3]], 1)
    # print(file_name)
    if file_name:
        cv2.imwrite(file_name, display)
