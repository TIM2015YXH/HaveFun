import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', default='data/hand_font_crop90.png', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--imgs_root', default='data/hand_font_crop90.png', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--size', default=512, type=int, help="output resolution")
    opt = parser.parse_args()

    # out_dir = os.path.dirname(opt.path)
    # out_rgba = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_rgba.png')
    # out_depth = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_depth.png')
    # out_normal = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_normal.png')
    # out_caption = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_caption.txt')

    names = sorted(os.listdir(opt.imgs_root))
    
    dst_root = os.path.join(opt.imgs_root, 'no_bkgd')
    os.makedirs(dst_root, exist_ok=True)
    print(dst_root)
    for name in tqdm(names):
        print(name)
        
        img_path = os.path.join(opt.imgs_root, name)

        # load image
        print(f'[INFO] loading image...')
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = image[...,-1] > 128
        image[~alpha] = np.array([0., 0., 0., 255])
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            cv2.imwrite(os.path.join(dst_root, 'xxx'+name), cv2.cvtColor(image, cv2.COLOR_RGB2BGRA))
            
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # carve background
        print(f'[INFO] background removal...')
        carved_image = BackgroundRemoval()(image) # [H, W, 4]
        mask = carved_image[..., -1] > 200
        mask = mask.astype(np.float32) * 255.
        carved_image[...,-1] = mask

        # # cv2.imwrite(out_rgba, cv2.cvtColor(carved_image, cv2.COLOR_RGBA2BGRA))
        # # cv2.imwrite('debug/mask.png', (mask.astype('uint8'))*255)
        cv2.imwrite(os.path.join(dst_root, name), cv2.cvtColor(carved_image, cv2.COLOR_RGB2BGRA))