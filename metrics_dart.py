import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import re
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips


def calculate_metrics(img_pred, img_gt, mask, flag, savepath, test_path, idx, lpips_model):

    image1 = cv2.imread(img_pred, cv2.IMREAD_UNCHANGED)[...,:3].astype(np.float64) / 255.
    image2 = cv2.imread(img_gt, cv2.IMREAD_UNCHANGED)[...,:3].astype(np.float64) / 255.
    mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)[...,0].astype(np.float64) / 255.

    # image1 = cv2.imread(img_pred, cv2.IMREAD_UNCHANGED)
    # image2 = cv2.imread(img_gt, cv2.IMREAD_UNCHANGED)[...,:3]
    # mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)[...,0]

    H, W, C = image1.shape
    

    
    choice = mask>0.5
    w, h, c = image1.shape
    alpha = (choice.astype(np.float64) * 1.).reshape(w,h,1)

    def calculate_psnr(image1, image2):
        mse = np.mean((image1 - image2) ** 2)
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    # psnr_value_masked = calculate_psnr(image1[choice], image2[choice])
    psnr_value_masked = calculate_psnr(image1[choice], image2[choice])
    # print("psnr_masked:", image1[choice].min(), image1[choice].max(), image2[choice].min(), image2[choice].max())
    image2[~choice] = np.array(1.)
    masked_pred = np.concatenate((image1, alpha), axis=-1)
    masked_gt = np.concatenate((image2, alpha), axis=-1)
    cv2.imwrite(os.path.join(test_path, f'masked_pred_{idx}.png'), (masked_pred*255.).astype(np.uint8))
    cv2.imwrite(os.path.join(test_path, f'masked_gt_{idx}.png'), (masked_gt*255.).astype(np.uint8))
    psnr_value_full = calculate_psnr(image1, image2)
    print()

    # print("ssim_value:", image1.min(), image1.max(), image2.min(), image2.max())
    ssim_value = ssim(image1, image2, multichannel=True)
    print()

    image1 = torch.from_numpy(image1).to(torch.float32).cuda()
    image2 = torch.from_numpy(image2).to(torch.float32).cuda()
    # image1_tensor = lpips.im2tensor(image1)
    # image2_tensor = lpips.im2tensor(image2)
    # lpips_value = lpips_model.forward(image1_tensor, image2_tensor).item()
    image1 = image1.reshape(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    image2 = image2.reshape(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    
    scaled_img1 = image1*2.-1.
    scaled_img2 = image2*2.-1.
   
    # print("lpips_value:", scaled_img1.min(), scaled_img1.max(), scaled_img2.min(), scaled_img2.max())
    lpips_value = lpips_model.forward(scaled_img1, scaled_img2).item()
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
    eval_imgs_root = os.path.join(pred, 'results_eval')
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
    txt_path = os.path.join(pred, 'metrics.txt')
    test_path = os.path.join(pred, 'verify_masked')
    os.makedirs(test_path, exist_ok=True)
    if os.path.exists(txt_path):
        os.remove(txt_path)
        
    
    lpips_model = lpips.LPIPS(net='vgg').cuda()
        
    
    for name in tqdm(img_names):
        idx = str(int(name.split('_')[1]))
        eval_img = os.path.join(eval_imgs_root, name)
        gt_img = os.path.join(gt_imgs, idx+'.png')
        gt_mask = os.path.join(gt_masks, idx+'.png')

        psnr_value_masked, psnr_value_full, ssim_value, lpips_value = calculate_metrics(eval_img, gt_img, gt_mask, name, txt_path, test_path, idx, lpips_model)
        psnr_m.append(psnr_value_masked)
        psnr_f.append(psnr_value_full)
        ssims.append(ssim_value)
        lpipss.append(lpips_value)
    mean_psnr_m = np.array(psnr_m).sum() / len(psnr_m)
    mean_psnr_f = np.array(psnr_f).sum() / len(psnr_f)
    mean_ssim = np.array(ssims).sum() / len(ssims)
    mean_lpips = torch.tensor(lpipss).sum() / len(lpipss)
    with open(txt_path, 'a+') as ff:
        ff.write(f'MEAN METRICS: ' + '\n')
        ff.write(f'psnr_value_masked: {mean_psnr_m:.2f} dB' + '\n')
        ff.write(f'psnr_value_full: {mean_psnr_f:.2f} dB' + '\n')
        ff.write(f'SSIM: {mean_ssim:.4f}' + '\n')
        ff.write(f'LPIPS: {mean_lpips:.4f}' + '\n')
        ff.write('\n')

