import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cv2
from manotorch.manolayer import ManoLayer
import vctoolkit as vc
from utils.draw3d import draw_aligned_mesh_plt
import pyrender
import trimesh
from htmlhand.utils.tools import get_mapping
from ikhand import skeletons
import torch


def display_pyrender(mesh):
    # cfg
    size = [480, 480]
    fov = 60
    trans = [0, 0, -0.3]

    # add cam and light
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[0, 0, 0])
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=size[0]/size[1])
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    scene.set_pose(node_cam, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    scene.add(light, pose=np.eye(4))

    # add mesh
    dt = np.array(trans)
    dt = dt[np.newaxis, :]
    mesh.vertices = mesh.vertices + dt
    m = pyrender.Mesh.from_trimesh(mesh)
    scene.add(m, pose=np.eye(4))
    # pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=size)# viewport_size=(1280, 768)

    # render
    r = pyrender.OffscreenRenderer(size[0], size[1])
    color, _ = r.render(scene)
    # cv2.imshow('test', color[..., ::-1])
    # cv2.waitKey(0)

    return color[..., ::-1]


if __name__ == '__main__':

    ik_file = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikmesh.json'
    save_dir = os.path.join( os.path.dirname(ik_file), ik_file.split('/')[-1].split('.')[0] )
    os.makedirs(save_dir, exist_ok=True)
    # K
    f = 1493
    c = [512, 512]
    cam_mat = np.array([
        [f, 0, c[0]],
        [0, f, c[1]],
        [0, 0, 1]
    ], np.float32)
    f = 1662.768
    c = [1080/2, 1920/2]
    cam_mat_ori = np.array([
        [f, 0, c[0]],
        [0, f, c[1]],
        [0, 0, 1]
    ], np.float32)
    # mesh template
    mesh = trimesh.load(os.path.join(os.path.dirname(__file__), '../htmlhand/vis.obj'), process=False)
    mapping = get_mapping(mesh.vertices)
    # Mano
    mano_layer = ManoLayer(
        rot_mode='axisang',
        use_pca=False,
        side='right',
        center_idx=0,
        mano_assets_root=os.path.join(os.path.dirname(__file__), '../template'),
        flat_hand_mean=True,
    )
    # mano_layer = ManoLayer(
    #     use_pca=False,
    #     side='right',
    #     center_idx=0,
    #     mano_root=os.path.join(os.path.dirname(__file__), '../template'),
    #     flat_hand_mean=True,
    # )
    # read data
    pose_dict_test = vc.load(ik_file)
    # render
    for i, sample in vc.progress_bar( enumerate( pose_dict_test.items() )):
        # if i<655:
        #     continue
        # read img
        image_path = sample[0]
        image_name = image_path.split('/')[-1]
        img = cv2.imread(image_path)
        # read
        root = np.array(sample[1]['root_ori'])
        verts_ori = np.array( sample[1]['verts_ori'] )
        theta = torch.tensor( sample[1]['theta'] ).view(1, -1)
        beta = torch.tensor( sample[1]['beta'] ).unsqueeze(0)
        scale = np.array( sample[1]['scale'] )
        rel_scale = np.array( sample[1]['rel_scale'] )
        camera_r = np.array( sample[1]['camera_r'] )
        camera_t = np.array( sample[1]['camera_t'] )
        mano_results = mano_layer(theta, beta)
        verts_mano = mano_results.verts[0].cpu().numpy() * scale
        joints_mano = mano_results.joints[0].cpu().numpy() * scale

        # global ori
        verts_ori_global = verts_ori + root
        # global IK
        verts_global = np.einsum('hw, vw -> vh', camera_r, verts_mano) + np.reshape(camera_t, [1, 3])
        verts_global[:, 1] *= -1
        # render
        mesh.vertices = (verts_mano - joints_mano[9])[mapping]
        render = display_pyrender(mesh)

        # plt
        plt_plot = img.copy()
        plt_plot = draw_aligned_mesh_plt(plt_plot, cam_mat, verts_global, mano_layer.th_faces, lw=2)
        plt_plot_ori = img.copy()
        plt_plot_ori = draw_aligned_mesh_plt(plt_plot_ori, cam_mat_ori, verts_ori_global, mano_layer.th_faces, lw=2)

        # display
        plt_plot = cv2.resize(plt_plot, (plt_plot.shape[1]//4, plt_plot.shape[0]//4))[..., :3]
        plt_plot_ori = cv2.resize(plt_plot_ori, (plt_plot_ori.shape[1]//4, plt_plot_ori.shape[0]//4))[..., :3]
        display = np.concatenate([plt_plot_ori, plt_plot, render], 1)
        # cv2.imshow('test', display)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, image_name), display)

