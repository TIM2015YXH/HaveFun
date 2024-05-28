import cv2
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src_root', type=str, required=False, default='real_hand/8717/images_crop', help="img root")
parser.add_argument('--mask_root', type=str, required=False, default='real_hand/8717/mobrecon_results_crop/mask', help="mask root")
parser.add_argument('--dst_root', type=str, required=False, default='real_hand_utils/no_wrist/images_no_wrist_crop_8717', help="save img root")

opt = parser.parse_args()



src_root = opt.src_root
img_name = sorted(os.listdir(src_root))
mask_root = opt.mask_root

dst_root = opt.dst_root

os.makedirs(dst_root, exist_ok=True)

for name in tqdm(img_name):
    mask_name = name.replace('jpg', 'png')
    img = cv2.imread(os.path.join(src_root, name), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_UNCHANGED)[...,0]
    mask = cv2.resize(mask, (512, 512),interpolation=cv2.INTER_NEAREST)
    choice = mask > 0
    alpha = choice.astype(np.uint8) * 255
    
    bottom_most_ones = 511 - np.argmax(alpha[::-1, :], axis=0)
    last = np.max(bottom_most_ones[200:300]) - 5
    # for i, k in enumerate(bottom_most_ones):
    #     if k>=1200:
    #         alpha[:k, i] = 255
    #         alpha[k:, i] = 0
    #     else:
    #         alpha[:k, i] = 255
    #         alpha[k:, i] = 255

    alpha[:last, :] = 255
    alpha[last:, :] = 0
        

    aff_crop_img_rgba = np.concatenate((img, alpha[...,None]), -1)
    cv2.imwrite(f'{dst_root}/{mask_name}', aff_crop_img_rgba)
    
    
