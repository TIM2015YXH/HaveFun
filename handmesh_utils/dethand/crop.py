import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
from utils.preprocessing import process_bbox, augmentation, augmentation_2d
import json
import torch

def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


if __name__ == '__main__':

    frames_dir = '/cpfs/shared/public/chenxingyu/few_shot_data/real_hand/8662/images_no_wrist'
    png_dir = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand/8662/images_no_wrist'
    crop_dir = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/no_wrist/' + 'crop'
    crop_withjoints_dir = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/no_wrist/' + 'crop_with_joints'
    joints_dir = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/no_wrist/' + 'joints_coords'
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(crop_withjoints_dir, exist_ok=True)
    bbox_path = frames_dir.split('/')[:-1] + ['bbox.pkl']
    bbox_path = '/' + os.path.join(*bbox_path)
    json_dir = frames_dir.split('/')[:-1] + ['mobrecon_results/json']
    json_dir = '/' + os.path.join(*json_dir)

    with open(bbox_path, 'rb') as f:
        bbox = pickle.load(f)

    for file_name in sorted(os.listdir(frames_dir)):
        print(file_name)
        if not os.path.splitext(file_name)[1][1:].lower() in ['jpg', 'png', 'jpeg']:
            continue
        key = os.path.join(frames_dir.replace('images_no_wrist', 'images'), file_name.replace('png', 'jpg'))
        png_key = os.path.join(png_dir, file_name)
        frame = cv2.imread(png_key, cv2.IMREAD_UNCHANGED)
        info = bbox[key]
        h, w = info['image_shape'][:2]
        box = info['instances'][0]['bbox']
        box[0] *= w
        box[2] *= w
        box[1] *= h
        box[3] *= h
        box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        box = process_bbox(box, w, h)
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, _ = augmentation(frame, box, 'eval',
                                                                                        exclude_flip=True,
                                                                                        input_img_shape=(512, 512), mask=None,
                                                                                        base_scale=1.3,
                                                                                        scale_factor=0.2,
                                                                                        rot_factor=0,
                                                                                        shift_wh=None,
                                                                                        gaussian_std=3,
                                                                                        bordervalue=(0,0,0))
        crop_name = os.path.join(crop_dir, file_name)
        
        b_name = os.path.basename(file_name).split(".")[0]
        jsonfile = os.path.join(json_dir, b_name+'.json')
        with open(jsonfile, 'rb') as ff:
            data = json.load(ff)
        # joints_uv = np.array(data['uv']).reshape(-1,2)
        # joints_uv[:,:1] *= w
        # joints_uv[:,1:2] *= h
        root = np.array(data['root'])
        K = np.array(data['K'])

        xyz_rel = np.array(data['xyz_rel']).reshape(-1, 3)
        
        focalx = K[0][0]
        focaly = K[1][1]
        
        extrinsic = np.array(
            [[ 1.0, 0.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0, 0.0],
            [ 0.0, 0.0, 1.0, 0.0],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
        )
        extrinsic[:3,3:] = root[...,None]

        
        joints_uvz = (K @ (extrinsic @ to_homogeneous(xyz_rel).T)[:3, :]).T
        joints_uvz[:,0:1] = joints_uvz[:,0:1] / joints_uvz[:,2:3]
        joints_uvz[:,1:2] = joints_uvz[:,1:2] / joints_uvz[:,2:3]
        joints_uv = joints_uvz[:,:2]
        # for coord in joints_uv:
        #     cv2.circle(frame, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 10, (0,0,255, 255), -1)
        # cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/test_ori_img', file_name), frame)
        
        print(box)
        joint_img, princpt = augmentation_2d(img, joints_uv, joints_uv, img2bb_trans, False)
        # _focalx = focalx * img.shape[1] / (box[2]*aug_param[1])
        # _focaly = focaly * img.shape[0] / (box[2]*aug_param[1])
        _focalx = _focaly = 1451.84814592
        print(_focalx)
        print(_focaly)
        # _focalx = focalx * img.shape[1] / (frame.shape[1]*aug_param[1])
        # _focaly = focaly * img.shape[0] / (frame.shape[0]*aug_param[1])
        # _focalx = focalx 
        # _focaly = focaly 
        
        dst_K = np.array(
            [
                [_focalx, 0., 256.],
                [0., _focaly, 256.],
                [0.,0.,1.]
            ]
        )

        dst_extrinsic = np.array(
            [[ 1.0, 0.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0, -0.07],
            [ 0.0, 0.0, 1.0, 3.2],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
        )
        
        dst_uvz = (dst_K @ (dst_extrinsic @ to_homogeneous((xyz_rel - xyz_rel[9])*5).T)[:3, :]).T
        # dst_uvz = (dst_K @ (dst_extrinsic @ to_homogeneous(xyz_rel).T)[:3, :]).T
        dst_uvz[:,0:1] = dst_uvz[:,0:1] / dst_uvz[:,2:3]
        dst_uvz[:,1:2] = dst_uvz[:,1:2] / dst_uvz[:,2:3]
        dst_uv = dst_uvz[:,:2]
        
        # idx1, idx2, idx3 = 0, 5, 13
        idx1, idx2, idx3 = 4, 8, 12
        
        src_joints_uv_toaff = np.zeros((3, 2), dtype=np.float32)

        src_joints_uv_toaff[0, :] = joint_img[idx1]
        src_joints_uv_toaff[1, :] = joint_img[idx2]
        src_joints_uv_toaff[2, :] = joint_img[idx3]

        dst_uv_toaff = np.zeros((3, 2), dtype=np.float32)

        dst_uv_toaff[0, :] = dst_uv[idx1]
        dst_uv_toaff[1, :] = dst_uv[idx2]
        dst_uv_toaff[2, :] = dst_uv[idx3]
        
        cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/crops', file_name), img)
        
        # for coord in joint_img:
        #     cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 0, 255, 255), -1)
        # for coord in dst_uvz:
        #     cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 255, 0, 255), -1)
        
        # cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/crops_withuv', file_name), img)
        
        
        trans = cv2.getAffineTransform(src_joints_uv_toaff, dst_uv_toaff)

        dst_crop_img = cv2.warpAffine(img, trans, (512,512))
        
        cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/affines', file_name), dst_crop_img)
        
        # cv2.imwrite(crop_name, img)
        # np.save(os.path.join(joints_dir, f'{b_name}.npy'), joint_img)
        # for coord in joint_img:
        #     cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 255, 0, 0), -1)
        # cv2.imwrite(os.path.join(crop_withjoints_dir, file_name), img)
            