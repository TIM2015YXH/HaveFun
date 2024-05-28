import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import re
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips

def calculate_metrics(img_pred, img_gt, mask, flag, savepath, test_path, idx):

    image1 = cv2.imread(img_pred, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(img_gt, cv2.IMREAD_UNCHANGED)[...,:3]
    mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)[...,0]
    
    choice = mask>128
    w, h, c = image1.shape
    alpha = (choice.astype(np.float32) * 255).reshape(w,h,1)

    def calculate_psnr(image1, image2):
        mse = np.mean((image1 - image2) ** 2)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    psnr_value_masked = calculate_psnr(image1[choice], image2[choice])
    image2[~choice] = np.array(255.)
    cv2.imwrite(os.path.join(test_path, f'masked_pred_{idx}.png'), np.concatenate((image1, alpha), axis=-1))
    cv2.imwrite(os.path.join(test_path, f'masked_gt_{idx}.png'), np.concatenate((image2, alpha), axis=-1))
    psnr_value_full = calculate_psnr(image1, image2)
    print()


    ssim_value = ssim(image1, image2, multichannel=True)
    print()


    lpips_model = lpips.LPIPS(net='vgg', version=0.1)  
    image1_tensor = lpips.im2tensor(image1)
    image2_tensor = lpips.im2tensor(image2)
    lpips_value = lpips_model.forward(image1_tensor, image2_tensor).item()
    print()

    with open(savepath, 'a+') as f:
        f.write(f'{flag}: ' + '\n')
        f.write(f'psnr_value_masked: {psnr_value_masked:.2f} dB' + '\n')
        f.write(f'psnr_value_full: {psnr_value_full:.2f} dB' + '\n')
        f.write(f'SSIM: {ssim_value:.4f}' + '\n')
        f.write(f'LPIPS: {lpips_value:.4f}' + '\n')
        f.write('\n')
    return psnr_value_masked, psnr_value_full, ssim_value, lpips_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True, help='root dir of GT 24 view images, e.g. /cpfs/shared/public/chenxingyu/few_shot_data/dart_24_views')
    parser.add_argument('--pred', type=str, required=True, help='the sub-dir of exp, e.g. out/exp/99')
    args = parser.parse_args()
    
    pred = args.pred
    sub = os.path.basename(pred)
    eval_imgs_root = os.path.join(pred, 'stage2', 'results_eval')
    gt_root = os.path.join(args.gt, f'{int(sub):03d}')
    gt_imgs = os.path.join(gt_root, 'basecolor')
    gt_masks = os.path.join(gt_root, 'mask')
    
    pattern = r'aeval_(\d+)_rgb\.png' 
    img_names = sorted(
    [filename for filename in os.listdir(eval_imgs_root) if re.search(pattern, filename)],
    key=lambda x: int(re.search(pattern, x).group(1))
    )

    psnr_m = []
    psnr_f = []
    ssims = []
    lpipss = []
    txt_path = os.path.join(pred, 'stage2', 'metrics.txt')
    test_path = os.path.join(pred, 'stage2', 'verify_masked')
    os.makedirs(test_path, exist_ok=True)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    for name in tqdm(img_names):
        idx = str(int(name.split('_')[1]))
        eval_img = os.path.join(eval_imgs_root, name)
        gt_img = os.path.join(gt_imgs, idx+'.png')
        gt_mask = os.path.join(gt_masks, idx+'.png')

        psnr_value_masked, psnr_value_full, ssim_value, lpips_value = calculate_metrics(eval_img, gt_img, gt_mask, name, txt_path, test_path, idx)
        psnr_m.append(psnr_value_masked)
        psnr_f.append(psnr_value_full)
        ssims.append(ssim_value)
        lpipss.append(lpips_value)
    mean_psnr_m = np.array(psnr_m).sum() / len(psnr_m)
    mean_psnr_f = np.array(psnr_f).sum() / len(psnr_f)
    mean_ssim = np.array(ssims).sum() / len(ssims)
    mean_lpips = np.array(lpipss).sum() / len(lpipss)
    with open(txt_path, 'a+') as ff:
        ff.write(f'MEAN METRICS: ' + '\n')
        ff.write(f'psnr_value_masked: {mean_psnr_m:.2f} dB' + '\n')
        ff.write(f'psnr_value_full: {mean_psnr_f:.2f} dB' + '\n')
        ff.write(f'SSIM: {mean_ssim:.4f}' + '\n')
        ff.write(f'LPIPS: {mean_lpips:.4f}' + '\n')
        ff.write('\n')

