import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import pickle
from utils.preprocessing import process_bbox, augmentation, augmentation_2d
import json
import torch
import argparse

def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--png_dir', type=str, required=False, default='few_shot_data/real_hand/8717/images_no_wrist_crop_8717', help="png root")
    parser.add_argument('--json_dir', type=str, required=False, default='few_shot_data/real_hand/8717/mobrecon_results_crop/json', help="json root")
    parser.add_argument('--front_id', type=str, required=False, default='000082')
    parser.add_argument('--back_id', type=str, required=False, default='000260')
    parser.add_argument('--xyz_rel_root', type=str, required=False, default='x', help="joints_root")
    parser.add_argument('--no_wrist_imgs_root', type=str, required=False, default='x', help="joints_root")
    parser.add_argument('--save_path', type=str, required=False, default='aligned_data', help="imgs save root")
    
    opt = parser.parse_args()

    png_dir = opt.png_dir
    json_dir = opt.json_dir
    save_path = opt.save_path
    # crop_dir = os.path.join(save_path, 'crop')
    # crop_withjoints_dir = os.path.join(save_path, 'crop_with_uv')
    # joints_dir = os.path.join(save_path, 'joints_coords')
    # os.makedirs(crop_dir, exist_ok=True)
    # os.makedirs(joints_dir, exist_ok=True)
    # os.makedirs(crop_withjoints_dir, exist_ok=True)
    h = w = 512

    for num in [opt.front_id, opt.back_id]:

        rot_x = np.array(
                    [[1,0,0],
                    [0,-1,0],
                    [0,0,-1]
                    ]
                )

        rot_y = np.array(
            [
                [-1,0,0],
                [0,1,0],
                [0,0,-1]
            ]
        )

        rot_z = np.array(
            [
                [-1,0,0],
                [0,-1,0],
                [0,0,1]
            ]
        )
        
        
        
        
        # for file_name in sorted(os.listdir(png_dir)):
        for file_name in [(f'{num}.png')]:
            print(file_name)
            if not os.path.splitext(file_name)[1][1:].lower() in ['jpg', 'png', 'jpeg']:
                continue

            png_key = os.path.join(png_dir, file_name)
            frame = cv2.imread(png_key, cv2.IMREAD_UNCHANGED)


            # crop_name = os.path.join(crop_dir, file_name)
            
            b_name = os.path.basename(file_name).split(".")[0]
            jsonfile = os.path.join(json_dir, b_name+'.json')
            with open(jsonfile, 'rb') as ff:
                data = json.load(ff)
            
            src_uv = np.array(data['uv']).reshape(-1,2)
            src_uv[:,:1] *= w
            src_uv[:,1:2] *= h

            K = np.array(data['K'])
            
            focalx = K[0][0]
            focaly = K[1][1]
            
            extrinsic = np.array(
                [[ 1.0, 0.0, 0.0, 0.0],
                [ 0.0, 1.0, 0.0, 0.0],
                [ 0.0, 0.0, 1.0, 0.0],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
            )
            # extrinsic[:3,3:] = root[...,None]

            
            # joints_uvz = (K @ (extrinsic @ to_homogeneous(xyz_rel).T)[:3, :]).T
            # joints_uvz[:,0:1] = joints_uvz[:,0:1] / joints_uvz[:,2:3]
            # joints_uvz[:,1:2] = joints_uvz[:,1:2] / joints_uvz[:,2:3]
            # joints_uv = joints_uvz[:,:2]
            # for coord in joints_uv:
            #     cv2.circle(frame, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 10, (0,0,255, 255), -1)
            # cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/test_ori_img', file_name), frame)
            
            # print(box)
            # joint_img, princpt = augmentation_2d(img, joints_uv, joints_uv, img2bb_trans, False)
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
            xyz_rel = np.load(os.path.join(opt.xyz_rel_root, f'x5-4_joints{num}_flip.npy'))
            dst_uvz = (dst_K @ (dst_extrinsic @ to_homogeneous(xyz_rel @ rot_x).T)[:3, :]).T
            # dst_uvz = (dst_K @ (dst_extrinsic @ to_homogeneous(xyz_rel).T)[:3, :]).T
            dst_uvz[:,0:1] = dst_uvz[:,0:1] / dst_uvz[:,2:3]
            dst_uvz[:,1:2] = dst_uvz[:,1:2] / dst_uvz[:,2:3]
            dst_uv = dst_uvz[:,:2]
            
            # idx1, idx2, idx3 = 0, 5, 13
            idx1, idx2, idx3 = 0, 3, 19
            
            src_joints_uv_toaff = np.zeros((3, 2), dtype=np.float32)

            src_joints_uv_toaff[0, :] = src_uv[idx1]
            src_joints_uv_toaff[1, :] = src_uv[idx2]
            src_joints_uv_toaff[2, :] = src_uv[idx3]

            dst_uv_toaff = np.zeros((3, 2), dtype=np.float32)

            dst_uv_toaff[0, :] = dst_uv[0]
            dst_uv_toaff[1, :] = dst_uv[15]
            dst_uv_toaff[2, :] = dst_uv[9]
            
            # cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/crops', file_name), img)
            
            # for coord in joint_img:
            #     cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 0, 255, 255), -1)
            # for coord in dst_uvz:
            #     cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 255, 0, 255), -1)
            
            # cv2.imwrite(os.path.join('/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/crops_withuv', file_name), img)
            
            
            trans = cv2.getAffineTransform(src_joints_uv_toaff, dst_uv_toaff)

            img = cv2.imread(os.path.join(opt.no_wrist_imgs_root, f'{num}.png'), cv2.IMREAD_UNCHANGED)
            dst_crop_img = cv2.warpAffine(img, trans, (512,512))
            
            # dstpath = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/real_hand_utils/test/affines_8717'
            dstpath = save_path
            os.makedirs(dstpath, exist_ok=True)
            cv2.imwrite(os.path.join(dstpath, file_name), dst_crop_img)
            
            # cv2.imwrite(crop_name, img)
            # np.save(os.path.join(joints_dir, f'{b_name}.npy'), joint_img)
            for coord in src_uv:
                print(coord)
                cv2.circle(img, (coord[0].astype(np.uint32),coord[1].astype(np.uint32)), 3, (0, 0, 255, 255), -1)
            # cv2.imwrite(os.path.join(crop_withjoints_dir, file_name), img)
            print(dst_uv)
            for coord in dst_uv:
                
                cv2.circle(img, (coord[0].astype(np.int32),coord[1].astype(np.int32)), 3, (0, 255, 0, 255), -1)
            # cv2.imwrite(os.path.join(crop_withjoints_dir, file_name), img)
                