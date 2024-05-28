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

get_Tpose = True
get_Apose = False
process_obj = True
camera_rotate = True
mesh_rotate = False
testing = True
training = True

identity = 24
case = 4
start = 1
step = 2
root_path = 'xavatar/training/{:03d}'.format(identity)
os.makedirs(os.path.join(root_path, 'basecolor'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'depth'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'mask'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'normal'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'smplx'), exist_ok=True)
# os.makedirs(os.path.join(root_path, 'mesh'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'gt'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'reference'), exist_ok=True)

# Set paths
DATA_DIR = "/cpfs/shared/public/chenxingyu/xavatar_dataset/{:05d}/train/Take{}".format(identity,case)
smplx_DIR = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.pkl".format(start))
with open(smplx_DIR, 'rb') as f:
    smplx_param = pickle.load(f)

print(smplx_param['global_orient'], type(smplx_param['global_orient']))

model_folder = 'model'
model = smplx.create(model_folder, model_type='smplx',
                         gender='male', use_face_contour=False, use_pca=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='pkl')

if testing:
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
    R_global = rodrigues(global_orient.unsqueeze(0).numpy())
    output = model(betas=betas, expression=expression, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                transl=transl, jaw_pose=jaw_pose,leye_pose=leye_pose,reye_pose=reye_pose, body_pose=body_pose, global_orient=global_orient,
                return_verts=True, return_full_pose=True)

    joints = output.joints.squeeze().to(device)
    obj_filename = os.path.join(DATA_DIR, "meshes_obj/mesh-f000{:02d}.obj".format(start))
    # smplx_obj_filename = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.ply".format(start))
    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    texture = mesh.textures

    scale = 0.50
    trans = -joints[0,:]
    verts = verts + torch.tensor(trans).to(device)
    verts = torch.bmm(verts.unsqueeze(0).float().to(device) , torch.tensor(R_global).float().to(device) ).squeeze(0)

    verts = verts.squeeze(0)*scale
    faces = faces
    tex = texture

    mesh = Meshes(
                verts=[verts.to(device)], faces=[faces.to(device)], textures=tex
            )
    test_path = 'xavatar/testing/{:03d}'.format(identity)
    os.makedirs(os.path.join(test_path, 'basecolor'), exist_ok=True)
    for i in range(-1,2,1):
        for j in range(8):
            print(i ,j)
            R, T = look_at_view_transform(dist=3.2, elev=i*45, azim=j*45, at=((0,0,0),))
            print(T)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20)
            raster_settings = RasterizationSettings(
                image_size=1024, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            lights = AmbientLights(device=device)

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
            image = images[0,...].cpu().numpy()
            image[...,:3] = image[...,:3]*255
            image[...,3] = (image[...,3]>0)*255
            depth = fragments.zbuf[0,...,0].cpu().numpy()
            mask = depth>-0.5
            depth[~mask] = 0
            depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
            depth = depth*255
            cv2.imwrite(os.path.join(test_path, 'basecolor', '{}.png'.format((i+1)*8+j)), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGRA) )
            # cv2.imwrite(os.path.join(test_path, 'depth', '{}.png'.format(idx)),depth.astype(np.uint8))

with open(os.path.join(root_path, 'smplx', 'mesh_smplx.pkl'), "wb") as f1:
    smplx_param['global_orient'] = np.zeros((1,3)).astype(np.float32)
    pickle.dump(smplx_param, f1)
    print(smplx_param)
    print(torch.tensor(smplx_param['global_orient'].reshape(-1,3)))

if get_Tpose:
    tpose_param = pickle.load(open('/cpfs/user/yangqianhe/stable-dreamfusion-latest/data/00016_variousposes2/tpose_smplx.pkl','rb'), encoding='Latent1')
    tpose_param['betas'] = smplx_param['betas']
    tpose_param['expression'] = smplx_param['expression']
    with open(os.path.join(root_path, 'smplx', 'tpose.pkl'), "wb") as ft:
        pickle.dump(tpose_param, ft)
        print(tpose_param)
    betas =torch.tensor(smplx_param['betas'].reshape(-1,10))
    expression = torch.tensor(smplx_param['expression'].reshape(-1,10))
    jaw_pose_T = torch.zeros((1,3))
    leye_pose_T = torch.zeros((1,3))
    reye_pose_T = torch.zeros((1,3))
    right_hand_pose_T = torch.zeros((1,45))
    left_hand_pose_T = torch.zeros((1,45))
    transl_T = torch.zeros((1,3))
    body_pose_T = torch.zeros((1,63))
    global_orient_T = torch.zeros((1,3))

    R_global = rodrigues(global_orient_T.unsqueeze(0).numpy())

    output_T = model(betas=betas, expression=expression, left_hand_pose=left_hand_pose_T, right_hand_pose=right_hand_pose_T,
                transl=transl_T, jaw_pose=jaw_pose_T,leye_pose=leye_pose_T,reye_pose=reye_pose_T, body_pose=body_pose_T, global_orient=global_orient_T,
                return_verts=True, return_full_pose=True)
    scale = 0.50
    joints_T = output_T.joints.squeeze().to(device)
    trans_T = -joints_T[0,:]

    SMPLX_verts_T = output_T.vertices.to(device)
    SMPLX_verts_T = SMPLX_verts_T + torch.tensor(trans_T).to(device)
    SMPLX_verts_T = torch.bmm(SMPLX_verts_T.float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 

    SMPLX_verts_T = SMPLX_verts_T*scale

    save_obj(os.path.join(root_path, 'smplx','tpose.obj'), verts=SMPLX_verts_T, faces=torch.Tensor(model.faces.astype(np.int16)).reshape(-1,3))

if get_Apose:
    betas =torch.tensor(smplx_param['betas'].reshape(-1,10))
    expression = torch.tensor(smplx_param['expression'].reshape(-1,10))
    jaw_pose_A = torch.zeros((1,3))
    leye_pose_A = torch.zeros((1,3))
    reye_pose_A = torch.zeros((1,3))
    right_hand_pose_A = torch.zeros((1,45))
    left_hand_pose_A = torch.zeros((1,45))
    transl_A = torch.zeros((1,3))
    body_pose_A = torch.zeros((1,63))
    global_orient_A = torch.zeros((1,3))

    R_global = rodrigues(global_orient.unsqueeze(0).numpy())

    output_A = model(betas=betas, expression=expression, left_hand_pose=left_hand_pose_A, right_hand_pose=right_hand_pose_A,
                transl=transl_A, jaw_pose=jaw_pose_A,leye_pose=leye_pose_A,reye_pose=reye_pose_A, body_pose=body_pose_A, global_orient=global_orient_A,
                return_verts=True, return_full_pose=True)

    scale = 0.50
    joints_A = output_A.joints.squeeze().to(device)
    trans_A = -joints_A[0,:]

    SMPLX_verts_A = output_A.vertices.to(device)
    SMPLX_verts_A = SMPLX_verts_A + torch.tensor(trans_A).to(device)
    SMPLX_verts_A = torch.bmm(SMPLX_verts_A.float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 

    SMPLX_verts_A = SMPLX_verts_A*scale

    save_obj('test/SMPLX_Apose.obj', verts=SMPLX_verts_A, faces=torch.Tensor(model.faces.astype(np.int16)).reshape(-1,3))


if training:
    for frame in range(start, start+step*8, step):
        idx = (frame - start)//step
        print('frame:',frame)
        smplx_DIR = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.pkl".format(frame))
        with open(smplx_DIR, 'rb') as f:
            smplx_param = pickle.load(f)

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

        # print(global_orient.unsqueeze(0))

        R_global = rodrigues(global_orient.unsqueeze(0).numpy())

        global_orient_aim = torch.zeros((1,3))
        full_pose = torch.cat([global_orient_aim.reshape(-1, 1, 3),
                                body_pose.reshape(-1, 21, 3),
                                jaw_pose.reshape(-1, 1, 3),
                                leye_pose.reshape(-1, 1, 3),
                                reye_pose.reshape(-1, 1, 3),
                                left_hand_pose.reshape(-1, 15, 3),
                                right_hand_pose.reshape(-1, 15, 3)],
                                dim=1).reshape(-1, 165)
        with open(os.path.join(root_path, 'smplx', '{}.pkl'.format(idx)), "wb") as f2:
            pickle.dump(full_pose, f2)

        # print(R_global, R_global.shape)


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

        if process_obj:
            
            obj_filename = os.path.join(DATA_DIR, "meshes_obj/mesh-f000{:02d}.obj".format(frame))
            smplx_obj_filename = os.path.join(DATA_DIR, "SMPLX/mesh-f000{:02d}_smplx.ply".format(frame))
            # Load obj file
            mesh = load_objs_as_meshes([obj_filename], device=device)

            verts = mesh.verts_packed()
            faces = mesh.faces_packed()
            texture = mesh.textures


            scale = 0.50
            trans = -joints[0,:]
            verts = verts + torch.tensor(trans).to(device)
            verts = torch.bmm(verts.unsqueeze(0).float().to(device) , torch.tensor(R_global).float().to(device) ).squeeze(0)

            verts = verts.squeeze(0)*scale
            faces = faces
            tex = texture

            #save transformed mesh to obj

            save_obj(os.path.join(root_path, 'gt','{}.obj'.format(idx)), verts=verts, faces=faces)

            # smplx_verts, smplx_faces = load_ply(smplx_obj_filename)
            # smplx_verts = smplx_verts.to(device) + torch.tensor(trans).to(device)
            # smplx_verts = torch.bmm(smplx_verts.unsqueeze(0).float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 
            # smplx_verts = smplx_verts*scale

            # # smplx_verts = torch.bmm(smplx_verts.unsqueeze(0).float() , torch.tensor(R_global).float() ).float() 
            # save_obj(os.path.join(root_path, 'mesh','{}.obj'.format(idx)), verts=smplx_verts, faces=smplx_faces)

            SMPLX_verts = output.vertices.to(device)
            SMPLX_verts = SMPLX_verts + torch.tensor(trans).to(device)
            SMPLX_verts = torch.bmm(SMPLX_verts.float() , torch.tensor(R_global).float().to(device) ).squeeze(0) 
            SMPLX_verts = SMPLX_verts*scale

            save_obj(os.path.join(root_path, 'smplx','{}.obj'.format(idx)), verts=SMPLX_verts, faces=torch.Tensor(model.faces.astype(np.int16)).reshape(-1,3))


            mesh = Meshes(
                        verts=[verts.to(device)], faces=[faces.to(device)], textures=tex
                    )


            hand_head_joints = hand_head_joints + torch.tensor(trans).to(device)
            hand_head_joints = torch.bmm(hand_head_joints.unsqueeze(0).float().to(device) , torch.tensor(R_global).float().to(device) ).squeeze(0)
            hand_head_verts = (hand_head_joints*scale).to(device)

            # Initialize a camera.
            # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
            # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
            # for i in range(8):
            if camera_rotate:
                R, T = look_at_view_transform(dist=3.2, elev=0.0, azim=idx*45, at=((0,0,0),))
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=20)
                raster_settings = RasterizationSettings(
                    image_size=1024, 
                    blur_radius=0.0, 
                    faces_per_pixel=1, 
                )
                lights = AmbientLights(device=device)

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
                image = images[0,...].cpu().numpy()
                image[...,:3] = image[...,:3]*255
                image[...,3] = (image[...,3]>0)*255
                depth = fragments.zbuf[0,...,0].cpu().numpy()
                mask = depth>-0.5
                depth[~mask] = 0
                depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
                depth = depth*255
                cv2.imwrite(os.path.join(root_path, 'basecolor', '{}.png'.format(idx)), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGRA) )
                cv2.imwrite(os.path.join(root_path, 'depth', '{}.png'.format(idx)),depth.astype(np.uint8))

                normal = phong_normal_shading(mesh, fragments)
                normal = normal[0,...,0,:3].cpu().numpy()
                normal[...,0] = normal[...,0]*(-1)
                normal[~mask] = 0
                normal[mask] = (normal[mask] - normal[mask].min()) / (normal[mask].max() - normal[mask].min() + 1e-9)
                normal = normal*255
                cv2.imwrite(os.path.join(root_path, 'normal', '{}.png'.format(str(idx))), cv2.cvtColor(normal.astype(np.uint8), cv2.COLOR_RGB2BGR) )
                mask = image[...,3]>0.5
                cv2.imwrite(os.path.join(root_path, 'mask', '{}.png'.format(str(idx))), (mask*255).astype(np.uint8))

                # project joints
                point_cloud = Pointclouds(points=[hand_head_verts], features=[rgb])
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

                cv2.imwrite(os.path.join(root_path, 'reference','{}.png'.format(str(idx))), cv2.cvtColor((smplx_joints_img*255).astype(np.uint8), cv2.COLOR_RGBA2BGRA) )

                face_kpt = np.argwhere(smplx_joints_img[...,2]>0)
                face_kpt_mean = face_kpt.mean(axis=0)/1024.0
                face_kpt_mean = [face_kpt_mean[1], face_kpt_mean[0]]

                left_hand_kpt = np.argwhere(smplx_joints_img[...,0]>0)
                left_hand_kpt_mean = left_hand_kpt.mean(axis=0)/1024.0
                left_hand_kpt_mean = [left_hand_kpt_mean[1], left_hand_kpt_mean[0]]

                right_hand_kpt = np.argwhere(smplx_joints_img[...,1]>0)                
                right_hand_kpt_mean = right_hand_kpt.mean(axis=0)/1024.0
                right_hand_kpt_mean = [right_hand_kpt_mean[1], right_hand_kpt_mean[0]]

                data = {"face_kpt_mean":face_kpt_mean,
                "hand_left_kpt_mean":left_hand_kpt_mean,
                "hand_right_kpt_mean":right_hand_kpt_mean,
                "scale": scale,
                "trans": trans.tolist(),
                "case": case,
                "start": start,
                "step": step,
                "frame": frame
                }
                out_path = os.path.join(root_path, 'smplx','{}'.format(str(idx)) + '_kpt_mean.json')
                with open(out_path, "w") as f2:
                    f2.write(json.dumps(data))

            # if mesh_rotate:

        
