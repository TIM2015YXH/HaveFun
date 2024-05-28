import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    PointsRasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    PointsRenderer, 
    MeshRasterizer,  
    PointsRasterizer, 
    SoftPhongShader,
    phong_normal_shading,
    TexturesUV,
    TexturesVertex, 
    AlphaCompositor
)
import numpy as np

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

import cv2
import pickle
import smplx
import json

def maxmin_normalize(x, eps=1e-20):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def clip_bbox(x, hw):
    return np.clip(x, 0, hw)

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R


case = 1
frame = 1
# Set paths
DATA_DIR = "/cpfs/user/yangqianhe/stable-dreamfusion-xchen/data/raw/00021/train/Take{}".format(case)
smplx_DIR = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.pkl".format(frame))
with open(smplx_DIR, 'rb') as f:
    smplx_param = pickle.load(f)

os.makedirs('test', exist_ok=True)

with open('test/mesh_smplx.pkl', "wb") as f1:
    pickle.dump(smplx_param, f1)

print(smplx_param)
model_folder = 'model'
model = smplx.create(model_folder, model_type='smplx',
                         gender='male', use_face_contour=False, use_pca=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='pkl')
print(model)
betas =torch.tensor(smplx_param['betas'].reshape(-1,10))
expression = torch.tensor(smplx_param['expression'].reshape(-1,10))
jaw_pose = torch.tensor(smplx_param['jaw_pose'].reshape(-1,3))
leye_pose = torch.tensor(smplx_param['leye_pose'].reshape(-1,3))
reye_pose = torch.tensor(smplx_param['reye_pose'].reshape(-1,3))
right_hand_pose = torch.tensor(smplx_param['right_hand_pose'].reshape(-1,45))
left_hand_pose = torch.tensor(smplx_param['left_hand_pose'].reshape(-1,45))
transl = torch.tensor(smplx_param['transl'].reshape(-1,3))
body_pose = torch.tensor(smplx_param['body_pose'].reshape(-1,63))
global_orient = torch.tensor(smplx_param['global_orient'].reshape(-1,3))
# joints = torch.tensor(smplx_param['joints'].reshape(-1,3))

print(global_orient.unsqueeze(0))

R_global = rodrigues(global_orient.unsqueeze(0).numpy())


print(R_global, R_global.shape)


output = model(betas=betas, expression=expression, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
            transl=transl, jaw_pose=jaw_pose,leye_pose=leye_pose,reye_pose=reye_pose, body_pose=body_pose, global_orient=global_orient,
            return_verts=True, return_full_pose=True)

joints = output.joints.squeeze().to(device)
left_hand_joints = joints[25:40,...] #15*3
right_hand_joints = joints[40:55,...] #15*3
head_joints = joints[76:127,...] #51*3

hand_head_joints = torch.cat((left_hand_joints,right_hand_joints,head_joints), 0)
rgb1 = np.tile(np.array([1,0,0,1]), (left_hand_joints.size(0),1))
rgb2 = np.tile(np.array([0,1,0,1]), (right_hand_joints.size(0),1))
rgb3 = np.tile(np.array([0,0,1,1]), (head_joints.size(0),1))
rgb = np.concatenate((rgb1,rgb2,rgb3), axis=0)

rgb = torch.Tensor(rgb).to(device)
# print(joints, joints.shape)

# print(output.vertices.shape)

print('transl', transl)

# SMPLX_verts = output.vertices.squeeze(0)

# save_obj('test/SMPLX_mesh.obj', verts=SMPLX_verts, faces=torch.Tensor(model.faces.astype(np.int16)).reshape(-1,3))

process_obj = False
if process_obj:
    obj_filename = os.path.join(DATA_DIR, "meshes_obj/mesh-f000{:02d}.obj".format(frame))
    smplx_obj_filename = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.ply".format(frame))
    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # plt.figure(figsize=(7,7))
    texture_image=mesh.textures.maps_padded()
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # plt.axis("off")

    # plt.figure(figsize=(7,7))
    # texturesuv_image_matplotlib(mesh.textures, subsample=None)
    # plt.axis("off")
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    texture = mesh.textures

    # verts = output.vertices
    # faces = model.faces
    # texture = 


    scale = 0.50
    # trans = [0,-1.15,0]
    trans = -joints[0,:]
    # print(mesh.verts_packed())
    # print(mesh.textures)
    verts = verts + torch.tensor(trans).to(device)
    verts = torch.bmm(verts.unsqueeze(0).float().to(device) , torch.tensor(R_global).float().to(device) ).squeeze(0) 
    # verts = verts + torch.tensor(trans).to(device)

    verts = verts.squeeze(0)*scale
    faces = faces
    tex = texture

    #save transformed mesh to obj

    # verts[..., 1] = verts[..., 1]-0.7

    save_obj('test/gt_mesh.obj', verts=verts, faces=faces)

    smplx_verts, smplx_faces = load_ply(smplx_obj_filename)
    smplx_verts = smplx_verts.to(device) + torch.tensor(trans).to(device)
    smplx_verts = torch.bmm(smplx_verts.unsqueeze(0).float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 
    # smplx_verts = smplx_verts + torch.tensor(trans)

    smplx_verts = smplx_verts*scale
    # print(smplx_verts.shape)
    # smplx_verts[...,1] = smplx_verts[...,1] - 0.7

    # smplx_verts = torch.bmm(smplx_verts.unsqueeze(0).float() , torch.tensor(R_global).float() ).float() 
    save_obj('test/smplx_mesh.obj', verts=smplx_verts, faces=smplx_faces)

    SMPLX_verts = output.vertices.to(device)
    SMPLX_verts = SMPLX_verts + torch.tensor(trans).to(device)
    SMPLX_verts = torch.bmm(SMPLX_verts.float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 

    

    SMPLX_verts = SMPLX_verts*scale

    save_obj('test/SMPLX_mesh.obj', verts=SMPLX_verts, faces=torch.Tensor(model.faces.astype(np.int16)).reshape(-1,3))

    

    mesh = Meshes(
                verts=[verts.to(device)], faces=[faces.to(device)], textures=tex
            )


    hand_head_joints = hand_head_joints + torch.tensor(trans).to(device)
    hand_head_joints = torch.bmm(hand_head_joints.unsqueeze(0).float().to(device) , torch.tensor(R_global).float().to(device) ).squeeze(0)
    

    smplx_verts = (hand_head_joints*scale).to(device)

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    for i in range(24):
        R, T = look_at_view_transform(dist=3.2, elev=0.0, azim=i*15, at=((0,0,0),))
        print(R)
        print(T)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        raster_settings = RasterizationSettings(
            image_size=1024, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        # lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        lights = AmbientLights(device=device)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )
        images, fragments = renderer(mesh)
        print(type(images))
        print(images.shape)
        image = images[0,...].cpu().numpy()
        print(image[...,3].max(),image[...,3].min())
        image[...,:3] = image[...,:3]*255
        image[...,3] = (image[...,3]>0)*255
        depth = fragments.zbuf[0,...,0].cpu().numpy()
        mask = depth>-0.5
        depth[~mask] = 0
        depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
        depth = depth*255
        print(depth.shape)
        print(np.min(depth),np.max(depth), depth[0,0])
        cv2.imwrite('test/00016_{}_rgba.png'.format(str(i)), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGRA) )
        cv2.imwrite('test/00016_{}_depth.png'.format(str(i)),depth.astype(np.uint8))


        normal = phong_normal_shading(mesh, fragments)
        print(type(normal))
        print(normal.shape)
        normal = normal[0,...,0,:3].cpu().numpy()
        normal[...,0] = normal[...,0]*(-1)
        normal[~mask] = 0
        normal[mask] = (normal[mask] - normal[mask].min()) / (normal[mask].max() - normal[mask].min() + 1e-9)
        normal = normal*255
        print(normal.shape)
        print(np.min(normal),np.max(normal), normal[0,0], normal[512,512])
        cv2.imwrite('test/00016_{}_normal.png'.format(str(i)), cv2.cvtColor(normal.astype(np.uint8), cv2.COLOR_RGB2BGR) )

        mask = image[...,3]>0.5
        cv2.imwrite('test/00016_{}_mask.png'.format(str(i)), (mask*255).astype(np.uint8))


        # project joints

        point_cloud = Pointclouds(points=[smplx_verts], features=[rgb])
        raster_settings2 = PointsRasterizationSettings(
            image_size=1024, 
            radius = 0.009,
            points_per_pixel = 1
        )
        rasterizer2 = PointsRasterizer(cameras=cameras, raster_settings=raster_settings2)
        renderer2 = PointsRenderer(
            rasterizer=rasterizer2,
            compositor=AlphaCompositor(background_color=(0, 0, 0))
            )
        smplx_joints_img = renderer2(point_cloud)
        smplx_joints_img = smplx_joints_img.detach().cpu().squeeze().numpy()
        print(smplx_joints_img.max(), smplx_joints_img.min())

        cv2.imwrite('test/00016_{}_smplx.png'.format(str(i)), cv2.cvtColor((smplx_joints_img*255).astype(np.uint8), cv2.COLOR_RGBA2BGRA) )

        face_kpt = np.argwhere(smplx_joints_img[...,2]>0)
        face_kpt_mean = face_kpt.mean(axis=0)/1024.0
        face_kpt_mean = [face_kpt_mean[1], face_kpt_mean[0]]


        left_hand_kpt = np.argwhere(smplx_joints_img[...,0]>0)
        print(left_hand_kpt)
        left_hand_kpt_mean = left_hand_kpt.mean(axis=0)/1024.0
        left_hand_kpt_mean = [left_hand_kpt_mean[1], left_hand_kpt_mean[0]]


        right_hand_kpt = np.argwhere(smplx_joints_img[...,1]>0)
        
        right_hand_kpt_mean = right_hand_kpt.mean(axis=0)/1024.0
        right_hand_kpt_mean = [right_hand_kpt_mean[1], right_hand_kpt_mean[0]]


        data = {"face_kpt_mean":face_kpt_mean,
        "hand_left_kpt_mean":left_hand_kpt_mean,
        "hand_right_kpt_mean":right_hand_kpt_mean,
        "scale": scale,
        '"trans': trans.tolist()
        }
        print(data)
        out_path = os.path.join('test/00016_{}'.format(str(i)) + '_kpt_mean.json')
        with open(out_path, "w") as f2:
            f2.write(json.dumps(data))



    