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

    ik_file = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikpose_conv_oriz.json'
    save_dir = os.path.join( os.path.dirname(ik_file), ik_file.split('/')[-1].split('.')[0] )
    os.makedirs(save_dir, exist_ok=True)
    # K
    f = 1493
    c = 512
    cam_mat = np.array([
        [f, 0, c],
        [0, f, c],
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
        verts = np.array( sample[1]['mano_verts'] )
        theta = torch.tensor( sample[1]['theta'] ).view(1, -1)
        beta = torch.tensor( sample[1]['beta'] ).unsqueeze(0)
        scale = np.array( sample[1]['scale'] )
        rel_scale = np.array( sample[1]['rel_scale'] )
        camera_r = np.array( sample[1]['camera_r'] )
        camera_t = np.array( sample[1]['camera_t'] )
        mano_results = mano_layer(theta, beta)
        verts_mano = mano_results.verts[0].cpu().numpy() * scale / rel_scale
        joints_mano = mano_results.joints[0].cpu().numpy() * scale / rel_scale

        # global
        # verts_global = np.einsum('hw, vw -> vh', camera_r, verts_mano) + np.reshape(camera_t, [1, 3])
        # verts_global[:, 2] *= -1
        # verts_global[:, 0] *= -1
        verts_global = verts_mano.copy()
        verts_global[:, 0:2] *= -1
        verts_global[:, 0] = verts_global[:, 0] + camera_t[0, 0]
        verts_global[:, 1] = verts_global[:, 1] - camera_t[0, 1]
        verts_global[:, 2] += camera_t[0, 2]
        # render
        # verts[:, 2] *= -1
        # verts[:, 0] *= -1
        mesh.vertices = (verts_mano - joints_mano[9])[mapping]
        render = display_pyrender(mesh)
        # uv
        uv = sample[1]['2d']
        uv_plot = vc.render_bones_from_uv(np.flip(uv, axis=-1).copy(), img.copy(), skeletons.KWAIHand)
        uv_plot = cv2.resize(uv_plot, (uv_plot.shape[1]//4, uv_plot.shape[0]//4))[..., :3]
        # plt
        plt_plot = img.copy()
        plt_plot = draw_aligned_mesh_plt(plt_plot, cam_mat, verts_global, mano_layer.th_faces, lw=2)
        plt_plot = cv2.resize(plt_plot, (plt_plot.shape[1]//4, plt_plot.shape[0]//4))[..., :3]
        # display
        display = np.concatenate([uv_plot, render, plt_plot], 1)
        # cv2.imshow('test', display)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, image_name), display)

