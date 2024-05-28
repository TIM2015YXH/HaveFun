import json
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm



th_hands_mean_right = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                        0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                        0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                        -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                        0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                        -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])
    

if __name__ == '__main__':
    annot_path = 'InterHand2.6M_test_MANO_NeuralAnnot.json'
    with open(annot_path, 'rb') as f:
        annot = json.load(f)

    annot = annot['0']

    poses = []
    for frame in tqdm(annot):
        if annot[frame]['right'] is None:
            print (frame)
            continue
        f_pose = annot[frame]['right']['pose']
        f_pose = np.array(f_pose)
        f_pose[3:] += th_hands_mean_right
        poses.append(f_pose)
    poses = np.array(poses).reshape(-1,16,3)
    poses = poses[150:,:,:]
    dst_path = '/cpfs/user/wangshaohui/merge_sdf/stable-dreamfusion/dmdrive/pose_data/interhand_poseseq_test.npy'
    np.save(dst_path, poses)
    print(poses.shape)
    print(th_hands_mean_right.shape)