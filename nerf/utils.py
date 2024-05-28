import os
import gc
import glob
import tqdm
import math
import imageio
import psutil
from pathlib import Path
import random
import shutil
import warnings
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchmetrics import PearsonCorrCoef

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import pickle

def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def maxmin_normalize(x, eps=1e-20):
    return (x - torch.min(x))/(torch.max(x) - torch.min(x) + eps)

def clip_bbox(x, hw):
    return np.clip(x, 0, hw).astype(np.int32)

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


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, head_ray=False, hand_ray=False, bbox=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''
    if (bbox is None) and head_ray:
        bbox = [W//2-W//8, 0]
    elif (bbox is None) and hand_ray:
        bbox = [W//2-W//16, 0]
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    if head_ray:
        i, j = custom_meshgrid(torch.linspace(bbox[0], bbox[0]+W//4-1, W//4, device=device), torch.linspace(bbox[1], bbox[1]+H//4-1, H//4, device=device))
        i = i.t().reshape([1, H//4*W//4]).expand([B, H//4*W//4]) + 0.5
        j = j.t().reshape([1, H//4*W//4]).expand([B, H//4*W//4]) + 0.5
    elif hand_ray:
        i, j = custom_meshgrid(torch.linspace(bbox[0], bbox[0]+W//8-1, W//8, device=device), torch.linspace(bbox[1], bbox[1]+H//8-1, H//8, device=device))
        i = i.t().reshape([1, H//8*W//8]).expand([B, H//8*W//8]) + 0.5
        j = j.t().reshape([1, H//8*W//8]).expand([B, H//8*W//8]) + 0.5
    else:
        i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
        i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
        j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(self,
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=10, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        if not opt.test:
            self.lpips = LPIPS(net_type='vgg').to(self.device)
        self.head_recon = opt.head_recon
        self.hand_recon = opt.hand_recon
        self.smplx_path = opt.smplx_path
        self.various_pose = opt.various_pose
        self.freeze_guidance = opt.freeze_guidance
        self.tpose = opt.tpose
        self.mesh_rotate = opt.mesh_rotate

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if self.opt.images is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            os.makedirs(os.path.join(self.workspace, 'head_temp'), exist_ok=True)
            os.makedirs(os.path.join(self.workspace, 'hand_temp'), exist_ok=True)

            # Save a copy of image_config in the experiment workspace
            if opt.image_config is not None:
                shutil.copyfile(opt.image_config, os.path.join(self.workspace, os.path.basename(opt.image_config)))

            # Save a copy of images in the experiment workspace
            if opt.images is not None:
                for image_file in opt.images:
                    shutil.copyfile(image_file, os.path.join(self.workspace, os.path.basename(image_file)))

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            if 'SD' in self.guidance:
                self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt.text])
                self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt.text])
                self.embeddings['IF']['uncond'] = self.guidance['IF'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'clip' in self.guidance:
                self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

        if self.opt.images is not None:

            h = int(self.opt.known_view_scale * self.opt.h)
            w = int(self.opt.known_view_scale * self.opt.w)

            # load processed image
            # for image in self.opt.images:
            #     assert image.endswith('_rgba.png') # the rest of this code assumes that the _rgba image has been passed.
            rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in self.opt.images]
            rgba_hw = np.stack([cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:])
            self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().to(self.device)
            self.mask = torch.from_numpy(rgba_hw[..., 3] > 0.5).to(self.device)

            # load depth
            if '_rgba.png' in self.opt.images[0]:
                depth_paths = [image.replace('_rgba.png', '_depth.png') for image in self.opt.images]
            else:
                depth_paths = [image.replace('basecolor', 'depth') for image in self.opt.images]
            depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
            depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
            if len(depth.shape)==4:
                depth = depth[..., 0]
            self.depth = torch.from_numpy(depth.astype(np.float32) / 255).to(self.device)  # TODO: this should be mapped to FP1

            # load normal   # TODO: don't load if normal loss is 0
            if '_rgba.png' in self.opt.images[0]:
                normal_paths = [image.replace('_rgba.png', '_normal.png') for image in self.opt.images]
            else:
                normal_paths = [image.replace('basecolor', 'normal') for image in self.opt.images]
                
            #### IMPORTANT: CONVERT COLOR ####
            normals = [cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) for normal_path in normal_paths]
            
            #################################
            for i in range(len(normals)):
                if normals[i].shape[-1] == 4:
                    normals[i] = normals[i][..., :3]
            normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
            self.normal = torch.from_numpy(normal.astype(np.float32) / 255).to(self.device)
            smplx_params = []
            mano_params = []

            if self.various_pose:
                if self.smplx_path is not None:
                    # smplx_paths = [image.replace('_rgba.png', '_smplx.pkl') for image in self.opt.images]
                    if '_rgba.png' in self.opt.images[0]:
                        smplx_paths = [image.replace('_rgba.png', '_smplx.pkl') for image in self.opt.images]
                    else:
                        smplx_paths = [image.replace('basecolor', 'smplx') for image in self.opt.images]
                        smplx_paths = [image.replace('.png', '.pkl') for image in smplx_paths]
                    for smplx_path in smplx_paths:
                        smplx_param = pickle.load(open(smplx_path, 'rb')) # [1,165] or [1,175]
                        smplx_params.append(smplx_param)
                    smplx_params = np.stack(smplx_params)
                    self.smplx_param = torch.from_numpy(smplx_params.astype(np.float32)).to(self.device)

                if self.opt.handy_path or self.opt.mano_path:   
                    if '_rgba.png' in self.opt.images[0]:
                        mano_param_paths = [image.replace('_rgba.png', '_mano_param.pth') for image in self.opt.images]
                    else:
                        mano_param_paths = [image.replace('basecolor', 'mano_param_flip') for image in self.opt.images]
                        mano_param_paths = [image.replace('.png', '.pth') for image in mano_param_paths]
                    for mano_param_path in mano_param_paths:
                        mano_param = torch.load(mano_param_path)
                        mano_param = torch.cat((mano_param[0], mano_param[1]), dim=1)
                        # print(mano_param)
                        mano_params.append(mano_param)
                    self.mano_param = torch.stack(mano_params).to(torch.float32).to(self.device)

            if self.head_recon>0:
                # kpt_paths = [image.replace('_rgba.png', '_kpt_mean.json') for image in self.opt.images]
                if '_rgba.png' in self.opt.images[0]:
                    kpt_paths = [image.replace('_rgba.png', '_kpt_mean.json') for image in self.opt.images]
                else:
                    kpt_paths = [image.replace('basecolor', 'bbox') for image in self.opt.images]
                    kpt_paths = [image.replace('.png', '_kpt_mean.json') for image in kpt_paths]
                ori_hw = 1024 if self.opt.dmtet else 512 # rgbas[0].shape[0]
                bbox_head_size = ori_hw//4
                rgbas = [cv2.resize(rgba, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA).astype(np.float32) for rgba in rgbas]
                depths = [cv2.resize(dh, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA) for dh in depths]
                normals = [cv2.resize(n, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA) for n in normals]

                rgbas_head = []
                depths_head = []
                normals_head = []
                head_bbox = []
                for i, r in enumerate(rgbas):
                    f = open(kpt_paths[i])
                    data_kpt = json.load(f)
                    face_kpt_mean = np.array(data_kpt['face_kpt_mean'])*ori_hw
                    face_bbox = face_kpt_mean.astype(np.int32) - bbox_head_size//2
                    face_bbox = clip_bbox(face_bbox, ori_hw-bbox_head_size)
                    # print(face_bbox)
                    head_bbox.append(face_bbox)
                    rgbas_head.append(r[face_bbox[1]:face_bbox[1]+bbox_head_size, face_bbox[0]:face_bbox[0]+bbox_head_size])
                    depths_head.append(depths[i][face_bbox[1]:face_bbox[1]+bbox_head_size, face_bbox[0]:face_bbox[0]+bbox_head_size].astype(np.float32))
                    normals_head.append(normals[i][face_bbox[1]:face_bbox[1]+bbox_head_size, face_bbox[0]:face_bbox[0]+bbox_head_size])               

                head_bbox = np.stack(head_bbox)
                # rgbas_head = [r[0:ori_hw//4, ori_hw//2-ori_hw//8:ori_hw//2-ori_hw//8+ori_hw//4] for r in rgbas_head]
                rgba_head_hw = np.stack([rh.astype(np.float32) / 255 for rh in rgbas_head])
                rgb_head_hw = rgba_head_hw[..., :3] * rgba_head_hw[..., 3:] + (1 - rgba_head_hw[..., 3:])
                mask_head_hw = rgba_head_hw[..., 3] > 0.5

                normal_head = np.stack(normals_head)

                #normalize depth
                for i,r in enumerate(depths_head):
                    depths_head[i][mask_head_hw[i,...]] = (r[mask_head_hw[i,...]]-np.min(r[mask_head_hw[i,...]])).astype(np.float32)/(np.max(r[mask_head_hw[i,...]]) - np.min(r[mask_head_hw[i,...]])+1e-6)
                depth_head = np.stack(depths_head)
                
                if self.opt.debug:

                    for i, r in enumerate(rgb_head_hw):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_head.png'), (r*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_mask.png'), (mask_head_hw[i,...]*255).astype(np.uint8))

                    for i, r in enumerate(depth_head):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_head_depth.png'), (r*255).astype(np.uint8))
                    for i, r in enumerate(normal_head):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_head_normal.png'), r)

                self.rgb_head = torch.from_numpy(rgb_head_hw).permute(0,3,1,2).contiguous().to(self.device)
                self.mask_head = torch.from_numpy(rgba_head_hw[..., 3] > 0.5).to(self.device)
                self.depth_head = torch.from_numpy(depth_head.astype(np.float32)).to(self.device)
                self.normal_head = torch.from_numpy(normal_head.astype(np.float32) / 255).to(self.device)
                self.bbox_head = head_bbox

            if self.hand_recon>0:
                ori_hw = 1024 if self.opt.dmtet else 1024 # rgbas[0].shape[0]
                bbox_hand_size = ori_hw//8
                rgbas = [cv2.resize(rgba, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA).astype(np.float32) for rgba in rgbas]
                depths = [cv2.resize(dh, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA) for dh in depths]
                normals = [cv2.resize(n, (ori_hw, ori_hw), interpolation=cv2.INTER_AREA) for n in normals]

                rgbas_hand_left = []
                depths_hand_left = []
                normals_hand_left = []
                rgbas_hand_right = []
                depths_hand_right = []
                normals_hand_right = []
                bbox_hand_left = []
                bbox_hand_right = []

                for i, r in enumerate(rgbas):
                    f = open(kpt_paths[i])
                    data_kpt = json.load(f)
                    hand_left_kpt_mean = np.array(data_kpt['hand_left_kpt_mean'])*ori_hw
                    hand_left_bbox = hand_left_kpt_mean.astype(np.int32) - bbox_hand_size//2
                    hand_left_bbox = clip_bbox(hand_left_bbox, ori_hw-bbox_hand_size)
                    # print(hand_left_bbox)
                    rgbas_hand_left.append(r[hand_left_bbox[1]:hand_left_bbox[1]+bbox_hand_size, hand_left_bbox[0]:hand_left_bbox[0]+bbox_hand_size])
                    depths_hand_left.append(depths[i][hand_left_bbox[1]:hand_left_bbox[1]+bbox_hand_size, hand_left_bbox[0]:hand_left_bbox[0]+bbox_hand_size].astype(np.float32))
                    normals_hand_left.append(normals[i][hand_left_bbox[1]:hand_left_bbox[1]+bbox_hand_size, hand_left_bbox[0]:hand_left_bbox[0]+bbox_hand_size])
                    bbox_hand_left.append(hand_left_bbox)

                    hand_right_kpt_mean = np.array(data_kpt['hand_right_kpt_mean'])*ori_hw
                    hand_right_bbox = hand_right_kpt_mean.astype(np.int32) - bbox_hand_size/2.0
                    hand_right_bbox = clip_bbox(hand_right_bbox, ori_hw-bbox_hand_size)
                    # print(hand_right_bbox)
                    # rgbas_hand_right = rgbas_hand_right.append(r[hand_right_bbox[1]:hand_right_bbox[1]+bbox_hand_size, hand_right_bbox[0]:hand_right_bbox[0]+bbox_hand_size])
                    rgbas_hand_right.append(r[hand_right_bbox[1]:hand_right_bbox[1]+bbox_hand_size, hand_right_bbox[0]:hand_right_bbox[0]+bbox_hand_size])
                    depths_hand_right.append(depths[i][hand_right_bbox[1]:hand_right_bbox[1]+bbox_hand_size, hand_right_bbox[0]:hand_right_bbox[0]+bbox_hand_size].astype(np.float32))
                    normals_hand_right.append(normals[i][hand_right_bbox[1]:hand_right_bbox[1]+bbox_hand_size, hand_right_bbox[0]:hand_right_bbox[0]+bbox_hand_size])
                    bbox_hand_right.append(hand_right_bbox)
                # left hand
                rgba_hand_left_hw = np.stack([rh.astype(np.float32) / 255 for rh in rgbas_hand_left])
                rgb_hand_left_hw = rgba_hand_left_hw[..., :3] * rgba_hand_left_hw[..., 3:] + (1 - rgba_hand_left_hw[..., 3:])
                mask_hand_left_hw = rgba_hand_left_hw[..., 3] > 0.5

                normal_hand_left = np.stack(normals_hand_left)

                bbox_hand_left = np.stack(bbox_hand_left)

                #normalize depth
                for i,r in enumerate(depths_hand_left):
                    depths_hand_left[i][mask_hand_left_hw[i,...]] = (r[mask_hand_left_hw[i,...]]-np.min(r[mask_hand_left_hw[i,...]])).astype(np.float32)/(np.max(r[mask_hand_left_hw[i,...]]) - np.min(r[mask_hand_left_hw[i,...]])+1e-6)
                depth_hand_left = np.stack(depths_hand_left)

                # right hand
                rgba_hand_right_hw = np.stack([rh.astype(np.float32) / 255 for rh in rgbas_hand_right])
                rgb_hand_right_hw = rgba_hand_right_hw[..., :3] * rgba_hand_right_hw[..., 3:] + (1 - rgba_hand_right_hw[..., 3:])
                mask_hand_right_hw = rgba_hand_right_hw[..., 3] > 0.5

                normal_hand_right = np.stack(normals_hand_right)
                bbox_hand_right = np.stack(bbox_hand_right)

                #normalize depth
                for i,r in enumerate(depths_hand_right):
                    depths_hand_right[i][mask_hand_right_hw[i,...]] = (r[mask_hand_right_hw[i,...]]-np.min(r[mask_hand_right_hw[i,...]])).astype(np.float32)/(np.max(r[mask_hand_right_hw[i,...]]) - np.min(r[mask_hand_right_hw[i,...]])+1e-6)
                depth_hand_right = np.stack(depths_hand_right)
                

                if self.opt.debug:
                    for i, r in enumerate(rgb_hand_left_hw):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_left.png'), (r*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_mask_hand_left.png'), (mask_hand_left_hw[i,...]*255).astype(np.uint8))
                    
                    for i, r in enumerate(depth_hand_left):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_left_depth.png'), (r*255).astype(np.uint8))
                    for i, r in enumerate(normal_hand_left):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_left_normal.png'), r)
                    
                    for i, r in enumerate(rgb_hand_right_hw):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_right.png'), (r*255).astype(np.uint8))
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_mask_hand_right.png'), (mask_hand_right_hw[i,...]*255).astype(np.uint8))
                    
                    for i, r in enumerate(depth_hand_right):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_right_depth.png'), (r*255).astype(np.uint8))
                    for i, r in enumerate(normal_hand_right):
                        cv2.imwrite(os.path.join(self.workspace, f'{i}_hand_right_normal.png'), r)
                
                self.rgb_hand_left = torch.from_numpy(rgb_hand_left_hw).permute(0,3,1,2).contiguous().to(self.device)
                self.mask_hand_left = torch.from_numpy(rgba_hand_left_hw[..., 3] > 0.5).to(self.device)
                self.depth_hand_left = torch.from_numpy(depth_hand_left.astype(np.float32)).to(self.device)
                self.normal_hand_left = torch.from_numpy(normal_hand_left.astype(np.float32) / 255).to(self.device)
                self.bbox_hand_left = bbox_hand_left

                self.rgb_hand_right = torch.from_numpy(rgb_hand_right_hw).permute(0,3,1,2).contiguous().to(self.device)
                self.mask_hand_right = torch.from_numpy(rgba_hand_right_hw[..., 3] > 0.5).to(self.device)
                self.depth_hand_right = torch.from_numpy(depth_hand_right.astype(np.float32)).to(self.device)
                self.normal_hand_right = torch.from_numpy(normal_hand_right.astype(np.float32) / 255).to(self.device)
                self.bbox_hand_right = bbox_hand_right

            print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

            print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

            print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

            # encode embeddings for zero123
            if 'zero123' in self.guidance:
                rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().to(self.device)
                guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
                self.embeddings['zero123']['default'] = {
                    'zero123_ws' : self.opt.zero123_ws,
                    'c_crossattn' : guidance_embeds[0],
                    'c_concat' : guidance_embeds[1],
                    'ref_polars' : self.opt.ref_polars,
                    'ref_azimuths' : self.opt.ref_azimuths,
                    'ref_radii' : self.opt.ref_radii,
                }
                if self.opt.lambda_normal_sds > 0:
                    normals_256 = np.stack([cv2.resize(__normal, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for __normal in normals])
                    normals_masked_256 = normals_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                    normal_256 = torch.from_numpy(normals_masked_256).permute(0,3,1,2).contiguous().to(self.device)
                    normal_embeds = self.guidance['zero123'].get_img_embeds(normal_256)
                    self.embeddings['zero123']['normal'] = {
                        'zero123_ws' : self.opt.zero123_ws,
                        'c_crossattn' : normal_embeds[0],
                        'c_concat' : normal_embeds[1],
                        'ref_polars' : self.opt.ref_polars,
                        'ref_azimuths' : self.opt.ref_azimuths,
                        'ref_radii' : self.opt.ref_radii,
                    }

            if 'clip' in self.guidance:
                self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)


    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------

    def train_step(self, data, save_guidance_path:Path=None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """
        # self.opt.debug = False

        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.images is not None and \
            (self.global_step % self.opt.known_view_interval == 0)

        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data
            if self.head_recon>0:
                rays_o_head = data['rays_o_head']
                rays_d_head = data['rays_d_head']  
            if self.hand_recon>0:
                rays_o_hand_left = data['rays_o_hand_left']
                rays_d_hand_left = data['rays_d_hand_left']
                rays_o_hand_right = data['rays_o_hand_right']
                rays_d_hand_right = data['rays_d_hand_right']             
            data['dst_pose'] = None

        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (self.opt.exp_end_iter - self.opt.exp_start_iter)

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0*exp_iter_ratio)
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                    self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                    self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + 2.0*exp_iter_ratio)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]
        poses = data['poses'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        if self.opt.pose_path is not None: # pose for animation
            dst_pose = data['dst_pose'] 
        else:
            dst_pose = None

        # find the closest view for zero123, if camera moves
        #if mesh moves, camera view is fixed as 0. Random select choice/poses to generate viewimage and zero123 
        # dst_pose = None
        if self.mesh_rotate: # random select a ref to render view image and zero123
            if self.various_pose and not do_rgbd_loss: 
                choice = torch.randperm(B)[:self.opt.batch_size]
                if choice>0 or self.tpose:
                    dst_pose = self.smplx_param[choice]
        else:
            if self.various_pose and not do_rgbd_loss: # select neart ref
                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']
                choice = self.guidance['zero123'].nearest_view(self.embeddings['zero123']['default'], polar, azimuth, radius)
                # if choice>0 or self.tpose:
                #     dst_pose = self.smplx_param[choice] # the base-pose to render images
                if self.tpose:
                    dst_pose = self.smplx_param[choice]
                elif self.opt.hand_tpose:
                    dst_pose = self.mano_param[choice]
                elif choice > 0:
                    if self.smplx_path is not None:
                        dst_pose = self.smplx_param[choice]
                    if self.opt.handy_path or self.opt.mano_path:
                        dst_pose = self.mano_param[choice]

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            use_hand = False
            
            if self.hand_recon>0:
                use_hand = True if np.random.rand(1)>0.5 else False
                use_left_hand = True if np.random.rand(1)>0.5 else False
            

            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]
            poses = poses[choice]

            if self.various_pose: # if choice is 0, do not deform
                if self.smplx_path is not None and (choice>0 or self.tpose):
                    dst_pose = self.smplx_param[choice] # the pose for each view image
                if (self.opt.handy_path or self.opt.mano_path) and (choice>0 or self.opt.hand_tpose):
                    dst_pose = self.mano_param[choice] # the pose for each view image
            else:
                dst_pose = None

            if do_rgbd_loss and self.head_recon>0:
                rays_o_head = rays_o_head[choice]
                rays_d_head = rays_d_head[choice]

                bbox_head = self.bbox_head[choice]
                if self.hand_recon>0:
                    rays_o_hand_left = rays_o_hand_left[choice]
                    rays_d_hand_left = rays_d_hand_left[choice]
                    rays_o_hand_right = rays_o_hand_right[choice]
                    rays_d_hand_right = rays_d_hand_right[choice]

                    bbox_hand_left = self.bbox_hand_left[choice]
                    bbox_hand_right = self.bbox_hand_right[choice]

        if do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False

            # bg_color_rand = torch.rand((1, 3)).to(self.device)
            # bg_color = bg_color_rand.repeat(B*N, 1)
            # bg_color_head = bg_color_rand.repeat(B*256*256, 1) if self.opt.dmtet else bg_color_rand.repeat(B*N, 1)
            bg_color = torch.rand((B * N, 3), device=rays_o.device)
            bg_color_head = torch.rand((B * 256 * 256, 3), device=rays_o.device) if self.opt.dmtet else torch.rand((B * N, 3), device=rays_o.device) 
            bg_color_hand = torch.rand((B * 128 * 128, 3), device=rays_o.device) if self.opt.dmtet else torch.rand((B * N, 3), device=rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale #* (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale
                if do_rgbd_loss and self.head_recon>0 and not use_hand:
                    rays_o_head = rays_o_head + torch.randn(3, device=self.device) * noise_scale
                    rays_d_head = rays_d_head + torch.randn(3, device=self.device) * noise_scale
                if do_rgbd_loss and self.hand_recon>0 and use_hand:
                    rays_o_hand_left = rays_o_hand_left + torch.randn(3, device=self.device) * noise_scale
                    rays_d_hand_left = rays_d_hand_left + torch.randn(3, device=self.device) * noise_scale
                    rays_o_hand_right = rays_o_hand_right + torch.randn(3, device=self.device) * noise_scale
                    rays_d_hand_right = rays_d_hand_right + torch.randn(3, device=self.device) * noise_scale

        elif exp_iter_ratio <= self.opt.latent_iter_ratio:
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                if self.opt.const_ambient_ratio > 0:
                    ambient_ratio = self.opt.const_ambient_ratio
                else:
                    ambient_ratio = self.opt.min_ambient_ratio + (1.0-self.opt.min_ambient_ratio) * random.random()
                # ambient_ratio = 1.0
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, poses=poses, 
                        staged=False, perturb=True, bg_color=bg_color, dst_pose=dst_pose,
                        ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        if do_rgbd_loss and self.head_recon>0 and not use_hand:
            HW_head = 1024 if self.opt.dmtet else np.sqrt(rays_o_head.shape[1]).astype('int')
            HW_head_crop = 256 if self.opt.dmtet else HW_head
            outputs_head = self.model.render(rays_o_head, rays_d_head, mvp, HW_head, HW_head, 
                                             poses=poses, staged=False, perturb=True, dst_pose=dst_pose, 
                                             bg_color=bg_color_head, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, head_recon=True, bbox = bbox_head)
            pred_depth_head = outputs_head['depth'].reshape(B, 1, HW_head_crop, HW_head_crop)
            pred_mask_head = outputs_head['weights_sum'].reshape(B, 1, HW_head_crop, HW_head_crop)

            if self.opt.debug:
                cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_depth.png'.format(self.global_step)), (pred_depth_head.reshape(B, HW_head_crop, HW_head_crop)[0].detach().cpu().numpy()*255).astype('uint8'))
                cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_mask.png'.format(self.global_step)), (pred_mask_head.reshape(B, HW_head_crop, HW_head_crop)[0].detach().cpu().numpy()*255).astype('uint8'))

            if 'normal_image' in outputs:
                pred_normal_head = outputs_head['normal_image'].reshape(B, HW_head_crop, HW_head_crop, 3)
                if self.opt.debug:
                    cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_normal.png'.format(self.global_step)), (pred_normal_head[0,...].detach().cpu().numpy()*255).astype('uint8'))
        if do_rgbd_loss and self.head_recon>0 and use_hand:
            HW_hand = 1024 if self.opt.dmtet else np.sqrt(rays_o_hand_left.shape[1]).astype('int')
            HW_hand_crop = 128 if self.opt.dmtet else HW_hand
            HW_hand_left_crop = HW_hand_right_crop = HW_hand_crop
            if use_left_hand:
                outputs_hand_left = self.model.render(rays_o_hand_left, rays_d_hand_left, mvp, HW_hand, HW_hand, 
                                                      poses=poses, staged=False, perturb=True, bg_color=bg_color_hand, dst_pose=dst_pose, 
                                                      ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, hand_recon=True, bbox = bbox_hand_left)
                pred_depth_hand_left = outputs_hand_left['depth'].reshape(B, 1, HW_hand_crop, HW_hand_crop)
                pred_mask_hand_left = outputs_hand_left['weights_sum'].reshape(B, 1, HW_hand_crop, HW_hand_crop)
                if self.opt.debug:
                    cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_left_depth.png'.format(self.global_step)), (pred_depth_hand_left.reshape(B, HW_hand_left_crop, HW_hand_left_crop)[0].detach().cpu().numpy()*255).astype('uint8'))
                    cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_left_mask.png'.format(self.global_step)), (pred_mask_hand_left.reshape(B, HW_hand_left_crop, HW_hand_left_crop)[0].detach().cpu().numpy()*255).astype('uint8'))

                if 'normal_image' in outputs:
                    pred_normal_hand_left = outputs_hand_left['normal_image'].reshape(B, HW_hand_crop, HW_hand_crop, 3)
                    if self.opt.debug:
                        cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_left_normal.png'.format(self.global_step)), (pred_normal_hand_left[0,...].detach().cpu().numpy()*255).astype('uint8'))
            else:
                outputs_hand_right = self.model.render(rays_o_hand_right, rays_d_hand_right, mvp, HW_hand, HW_hand, 
                                                       poses=poses, staged=False, perturb=True, bg_color=bg_color_hand, dst_pose=dst_pose, 
                                                       ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, hand_recon=True, bbox = bbox_hand_right)
                pred_depth_hand_right = outputs_hand_right['depth'].reshape(B, 1, HW_hand_crop, HW_hand_crop)
                pred_mask_hand_right = outputs_hand_right['weights_sum'].reshape(B, 1, HW_hand_crop, HW_hand_crop)
                if self.opt.debug:
                    cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_right_depth.png'.format(self.global_step)), (pred_depth_hand_right.reshape(B, HW_hand_right_crop, HW_hand_right_crop)[0].detach().cpu().numpy()*255).astype('uint8'))
                    cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_right_mask.png'.format(self.global_step)), (pred_mask_hand_right.reshape(B, HW_hand_right_crop, HW_hand_right_crop)[0].detach().cpu().numpy()*255).astype('uint8'))

                if 'normal_image' in outputs:
                    pred_normal_hand_right = outputs_hand_right['normal_image'].reshape(B, HW_hand_crop, HW_hand_crop, 3)
                    if self.opt.debug:
                        cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_right_normal.png'.format(self.global_step)), (pred_normal_hand_right[0,...].detach().cpu().numpy()*255).astype('uint8'))

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            if self.opt.debug:
                cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_body_rgb.png'.format(self.global_step)), (pred_rgb[0].detach().permute(1,2,0).cpu().numpy()*255).astype('uint8'))

            if do_rgbd_loss and self.head_recon>0 and not use_hand:
                pred_rgb_head = outputs_head['image'].reshape(B, HW_head_crop, HW_head_crop, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
                if self.opt.debug:
                    cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_rgb.png'.format(self.global_step)), (pred_rgb_head[0].detach().permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            
            if do_rgbd_loss and self.hand_recon>0 and use_hand:
                if use_left_hand:
                    pred_rgb_hand_left = outputs_hand_left['image'].reshape(B, HW_hand_crop, HW_hand_crop, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
                    if self.opt.debug:
                        cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_left_rgb.png'.format(self.global_step)), (pred_rgb_hand_left[0].detach().permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                else:
                    pred_rgb_hand_right = outputs_hand_right['image'].reshape(B, HW_hand_crop, HW_hand_crop, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
                    if self.opt.debug:
                        cv2.imwrite(os.path.join(self.workspace, 'hand_temp/{}_right_rgb.png'.format(self.global_step)), (pred_rgb_hand_right[0].detach().permute(1,2,0).cpu().numpy()*255).astype('uint8'))


        # known view loss
        loss_rgb = loss_normal = loss_depth = loss_mask = loss_guidance = torch.tensor(-1)
        loss_rgb_head = loss_normal_head = loss_depth_head = loss_mask_head = torch.tensor(-1)
        loss_rgb_hand_left = loss_normal_hand_left = loss_depth_hand_left = loss_mask_hand_left = torch.tensor(-1)
        loss_rgb_hand_right = loss_normal_hand_right = loss_depth_hand_right = loss_mask_hand_right = torch.tensor(-1)
        if do_rgbd_loss:
            if choice==0:
                lambda_head_recon = 2*self.head_recon
            else:
                lambda_head_recon = self.head_recon
            gt_mask = self.mask # [B, H, W]
            gt_rgb = self.rgb   # [B, 3, H, W]
            gt_normal = self.normal # [B, H, W, 3]
            gt_depth = self.depth   # [B, H, W]
            if self.head_recon>0 and not use_hand:
                gt_mask_head = self.mask_head # [B, H, W]
                gt_rgb_head = self.rgb_head   # [B, 3, H, W]
                gt_normal_head = self.normal_head # [B, H, W, 3]
                gt_depth_head = self.depth_head   # [B, H, W]
            if self.hand_recon>0 and use_hand:
                gt_mask_hand_left = self.mask_hand_left # [B, H, W]
                gt_rgb_hand_left = self.rgb_hand_left   # [B, 3, H, W]
                gt_normal_hand_left = self.normal_hand_left # [B, H, W, 3]
                gt_depth_hand_left = self.depth_hand_left   # [B, H, W]
                gt_mask_hand_right = self.mask_hand_right # [B, H, W]
                gt_rgb_hand_right = self.rgb_hand_right   # [B, 3, H, W]
                gt_normal_hand_right = self.normal_hand_right # [B, H, W, 3]
                gt_depth_hand_right = self.depth_hand_right   # [B, H, W]

            if len(gt_rgb) > self.opt.batch_size:
                gt_mask = gt_mask[choice]
                gt_rgb = gt_rgb[choice]
                gt_normal = gt_normal[choice]
                gt_depth = gt_depth[choice]
                if self.head_recon>0 and not use_hand:
                    gt_mask_head = gt_mask_head[choice]
                    gt_rgb_head = gt_rgb_head[choice]
                    gt_normal_head = gt_normal_head[choice]
                    gt_depth_head = gt_depth_head[choice]
                if self.hand_recon>0 and use_hand:
                    gt_mask_hand_left = gt_mask_hand_left[choice]
                    gt_rgb_hand_left = gt_rgb_hand_left[choice]
                    gt_normal_hand_left = gt_normal_hand_left[choice]
                    gt_depth_hand_left = gt_depth_hand_left[choice]
                    gt_mask_hand_right = gt_mask_hand_right[choice]
                    gt_rgb_hand_right = gt_rgb_hand_right[choice]
                    gt_normal_hand_right = gt_normal_hand_right[choice]
                    gt_depth_hand_right = gt_depth_hand_right[choice]

            # color loss
            gt_rgb = gt_rgb * gt_mask[:, None].float() + bg_color.reshape(B, H, W, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask[:, None].float())
            # cv2.imwrite('check_mesh/gt1.png', (gt_rgb[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
            # cv2.imwrite('check_mesh/pred_rgb1.png', (pred_rgb[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
            loss_rgb = F.mse_loss(pred_rgb, gt_rgb) + self.lpips(pred_rgb*2-1, gt_rgb*2-1)
            loss = self.opt.lambda_rgb * loss_rgb
            if self.opt.debug:
                cv2.imwrite(os.path.join(self.workspace, 'head_temp/{}_gt_rgb.png'.format(self.global_step)), (gt_rgb[0].detach().permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            if self.head_recon>0 and not use_hand:
                gt_rgb_head = gt_rgb_head * gt_mask_head[:, None].float() + bg_color_head.reshape(B, HW_head_crop, HW_head_crop, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask_head[:, None].float())
                loss_rgb_head = F.mse_loss(pred_rgb_head, gt_rgb_head) + self.lpips(pred_rgb_head*2-1, gt_rgb_head*2-1)
                loss = loss + self.opt.lambda_rgb * loss_rgb_head * lambda_head_recon
            
            if self.hand_recon>0 and use_hand:
                if use_left_hand:
                    gt_rgb_hand_left = gt_rgb_hand_left * gt_mask_hand_left[:, None].float() + bg_color_hand.reshape(B, HW_hand_crop, HW_hand_crop, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask_hand_left[:, None].float())
                    loss_rgb_hand_left = F.mse_loss(pred_rgb_hand_left, gt_rgb_hand_left) + self.lpips(pred_rgb_hand_left*2-1, gt_rgb_hand_left*2-1)
                    loss = loss + self.opt.lambda_rgb * loss_rgb_hand_left * self.hand_recon
                else:
                    gt_rgb_hand_right = gt_rgb_hand_right * gt_mask_hand_right[:, None].float() + bg_color_hand.reshape(B, HW_hand_crop, HW_hand_crop, 3).permute(0,3,1,2).contiguous() * (1 - gt_mask_hand_right[:, None].float())
                    loss_rgb_hand_right = F.mse_loss(pred_rgb_hand_right, gt_rgb_hand_right) + self.lpips(pred_rgb_hand_right*2-1, gt_rgb_hand_right*2-1)
                    loss = loss + self.opt.lambda_rgb * loss_rgb_hand_right * self.hand_recon
            
            # lambda_mask = self.opt.lambda_mask
            # mask loss
            # if choice==0:
            #     lambda_mask = 2*self.opt.lambda_mask
            # else:
            #     lambda_mask = self.opt.lambda_mask
            lambda_mask = self.opt.lambda_mask
            loss_mask =  F.mse_loss(pred_mask[:, 0], gt_mask.float())
            loss = loss + lambda_mask * loss_mask
            if self.head_recon>0 and not use_hand:
                loss_mask_head =  F.mse_loss(pred_mask_head[:, 0], gt_mask_head.float())
                loss = loss + lambda_mask * loss_mask_head * lambda_head_recon
            
            if self.hand_recon>0 and use_hand:
                if use_left_hand:
                    loss_mask_hand_left =  F.mse_loss(pred_mask_hand_left[:, 0], gt_mask_hand_left.float())
                    loss = loss + lambda_mask * loss_mask_hand_left * self.hand_recon
                else:
                    loss_mask_hand_right =  F.mse_loss(pred_mask_hand_right[:, 0], gt_mask_hand_right.float())
                    loss = loss + lambda_mask * loss_mask_hand_right * self.hand_recon

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs:
                # valid_gt_normal = 1 - 2 * gt_normal[gt_mask] # [B, 3]
                valid_gt_normal = 2 * gt_normal[gt_mask] - 1 # [B, 3]
                valid_pred_normal = 2 * pred_normal[gt_mask] - 1 # [B, 3]
                # if not self.opt.dmtet:
                #     lambda_normal = self.opt.lambda_normal * min(1, self.global_step  / self.opt.iters)
                # else:
                #     lambda_normal = self.opt.lambda_normal * min(1, 10*self.global_step  / self.opt.iters)
                lambda_normal = self.opt.lambda_normal
                loss_normal = 1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean()
                loss = loss + lambda_normal * loss_normal
                if self.head_recon>0 and not use_hand:
                    valid_gt_normal_head = 2 * gt_normal_head[gt_mask_head] - 1 # [B, 3] why 1-2*normal
                    valid_pred_normal_head = 2 * pred_normal_head[gt_mask_head] - 1 # [B, 3] why 1-2*normal
                    loss_normal_head = 1 - F.cosine_similarity(valid_pred_normal_head, valid_gt_normal_head).mean()
                    loss = loss + lambda_normal * loss_normal_head * lambda_head_recon
                
                if self.hand_recon>0 and use_hand:
                    if use_left_hand:
                        valid_gt_normal_hand_left = 2 * gt_normal_hand_left[gt_mask_hand_left] - 1 # [B, 3] why 1-2*normal
                        valid_pred_normal_hand_left = 2 * pred_normal_hand_left[gt_mask_hand_left] - 1 # [B, 3] why 1-2*normal
                        loss_normal_hand_left = 1 - F.cosine_similarity(valid_pred_normal_hand_left, valid_gt_normal_hand_left).mean()
                        loss = loss + lambda_normal * loss_normal_hand_left * self.hand_recon
                    else:
                        valid_gt_normal_hand_right = 2 * gt_normal_hand_right[gt_mask_hand_right] - 1 # [B, 3] why 1-2*normal
                        valid_pred_normal_hand_right = 2 * pred_normal_hand_right[gt_mask_hand_right] - 1 # [B, 3] why 1-2*normal
                        loss_normal_hand_right = 1 - F.cosine_similarity(valid_pred_normal_hand_right, valid_gt_normal_hand_right).mean()
                        loss = loss + lambda_normal * loss_normal_hand_right * self.hand_recon

            # relative depth loss
            if self.opt.lambda_depth > 0:
                valid_gt_depth = gt_depth[gt_mask] # [B,]
                valid_pred_depth = pred_depth[:, 0][gt_mask] # [B,]
                # if not self.opt.dmtet:
                #     lambda_depth = self.opt.lambda_depth * min(1, self.global_step / self.opt.iters)
                # else:
                #     lambda_depth = self.opt.lambda_depth * min(1, 10*self.global_step  / self.opt.iters)
                lambda_depth = self.opt.lambda_depth
                loss_depth = 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                loss = loss + lambda_depth * loss_depth
                if self.head_recon>0 and not use_hand:
                    valid_gt_depth_head = gt_depth_head[gt_mask_head] # [B,]
                    valid_pred_depth_head = pred_depth_head[:, 0][gt_mask_head] # [B,]
                    loss_depth_head = 1 - self.pearson(valid_pred_depth_head, valid_gt_depth_head) if torch.std(valid_pred_depth_head)>0 else torch.tensor(2.0).to(self.device)
                    loss = loss + lambda_depth * loss_depth_head * lambda_head_recon

                if self.hand_recon>0 and use_hand:
                    if use_left_hand:
                        valid_gt_depth_hand_left = gt_depth_hand_left[gt_mask_hand_left] # [B,]
                        valid_pred_depth_hand_left = pred_depth_hand_left[:, 0][gt_mask_hand_left] # [B,]
                        loss_depth_hand_left = 1 - self.pearson(valid_pred_depth_hand_left, valid_gt_depth_hand_left) if torch.std(valid_pred_depth_hand_left)>0 else torch.tensor(2.0).to(self.device)
                        loss = loss + lambda_depth * loss_depth_hand_left * self.hand_recon
                    else:
                        valid_gt_depth_hand_right = gt_depth_hand_right[gt_mask_hand_right] # [B,]
                        valid_pred_depth_hand_right = pred_depth_hand_right[:, 0][gt_mask_hand_right] # [B,]
                        loss_depth_hand_right = 1 - self.pearson(valid_pred_depth_hand_right, valid_gt_depth_hand_right) if torch.std(valid_pred_depth_hand_right)>0 else torch.tensor(2.0).to(self.device)
                        loss = loss + lambda_depth * loss_depth_hand_right * self.hand_recon

                # # scale-invariant
                # with torch.no_grad():
                #     A = torch.cat([valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1) # [B, 2]
                #     X = torch.linalg.lstsq(A, valid_pred_depth).solution # [2, 1]
                #     valid_gt_depth = A @ X # [B, 1]
                # lambda_depth = self.opt.lambda_depth #* min(1, self.global_step / self.opt.iters)
                # loss = loss + lambda_depth * F.mse_loss(valid_pred_depth, valid_gt_depth)

        elif self.freeze_guidance:# do not leverage guidance
            loss = 0
        # novel view loss
        else:
            loss = 0
                    
            if 'zero123' in self.guidance:

                lambda_sds = self.opt.lambda_sds
                
                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                if self.various_pose:
                    if self.opt.denormalize:
                        embedding = {
                        'zero123_ws' : torch.tensor(self.embeddings['zero123']['default']['zero123_ws']),
                        'c_crossattn' : self.embeddings['zero123']['default']['c_crossattn'],
                        'c_concat' : self.embeddings['zero123']['default']['c_concat'],
                        'ref_polars' : torch.tensor(self.embeddings['zero123']['default']['ref_polars']),
                        'ref_azimuths' : torch.tensor(self.embeddings['zero123']['default']['ref_azimuths']),
                        'ref_radii' : torch.tensor(self.embeddings['zero123']['default']['ref_radii']),
                    }
                        loss_guidance = self.guidance['zero123'].train_step(embedding, pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)
                    else:
                        embedding = {
                        'zero123_ws' : torch.tensor(self.embeddings['zero123']['default']['zero123_ws'][choice]).reshape(1),
                        'c_crossattn' : self.embeddings['zero123']['default']['c_crossattn'][choice],
                        'c_concat' : self.embeddings['zero123']['default']['c_concat'][choice],
                        'ref_polars' : torch.tensor(self.embeddings['zero123']['default']['ref_polars'][choice]).reshape(1),
                        'ref_azimuths' : torch.tensor(self.embeddings['zero123']['default']['ref_azimuths'][choice]).reshape(1),
                        'ref_radii' : torch.tensor(self.embeddings['zero123']['default']['ref_radii'][choice]).reshape(1),
                    }
                        loss_guidance = self.guidance['zero123'].train_step(embedding, pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                    as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)
                else:
                    loss_guidance = self.guidance['zero123'].train_step(self.embeddings['zero123']['default'], pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                    as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)

                # loss_guidance = self.guidance['zero123'].train_step(self.embeddings['zero123']['default'], pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale, as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)
                if 'normal_image' in outputs and self.opt.lambda_normal_sds > 0:
                    __pred_normal = pred_normal.permute(0,3,1,2).contiguous().to(self.device)
                    loss_normal_guidance = self.guidance['zero123'].train_step(self.embeddings['zero123']['normal'], __pred_normal, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                    as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path)
                    # loss = loss + loss_guidance
                    loss = loss + lambda_sds * loss_guidance + self.opt.lambda_normal_sds * loss_normal_guidance
                else:
                    loss = loss + lambda_sds * loss_guidance

        # regularizations
        loss_mesh_laplacian = loss_mesh_normal = loss_opacity = loss_entropy = loss_smooth = loss_orient = loss_normal_perturb = torch.tensor(-1)
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)
                loss = loss + lambda_entropy * loss_entropy

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
                # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
                # total-variation
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_normal_perturb = outputs['loss_normal_perturb']
                loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb

        else:

            if self.opt.lambda_mesh_normal > 0:
                loss_mesh_normal = self.opt.lambda_mesh_normal * outputs['normal_loss']
                loss = loss + loss_mesh_normal

            if self.opt.lambda_mesh_laplacian > 0:
                loss_mesh_laplacian = self.opt.lambda_mesh_laplacian * outputs['lap_loss']
                loss = loss + loss_mesh_laplacian

        return pred_rgb, pred_depth, loss, {'loss_rgb': loss_rgb, 'loss_normal': loss_normal, 'loss_depth': loss_depth, 'loss_mask': loss_mask,
                                            'loss_mesh_normal': loss_mesh_normal, 'loss_mesh_laplacian': loss_mesh_laplacian,
                                            'loss_opacity': loss_opacity,
                                            'loss_entropy': loss_entropy,
                                            'loss_smooth': loss_smooth,
                                            'loss_orient': loss_orient,
                                            'loss_normal_perturb': loss_normal_perturb,
                                            'loss_guidance': loss_guidance,
                                            'loss_rgb_head': loss_rgb_head, 'loss_normal_head': loss_normal_head, 'loss_depth_head': loss_depth_head, 'loss_mask_head': loss_mask_head,
                                            'loss_rgb_hand_left': loss_rgb_hand_left, 'loss_normal_hand_left': loss_normal_hand_left, 'loss_depth_hand_left': loss_depth_hand_left, 'loss_mask_hand_left': loss_mask_hand_left,
                                            'loss_rgb_hand_right': loss_rgb_hand_right, 'loss_normal_hand_right': loss_normal_hand_right, 'loss_depth_hand_right': loss_depth_hand_right, 'loss_mask_hand_right': loss_mask_hand_right,
                                            }

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']
        poses = data['poses']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, 
                                    poses=poses, staged=True, perturb=False, bg_color=None,
                                    light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W)
        pred_normal = None
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        # pred_rgb_head = pred_depth_head = pred_normal_head = None
        # if self.head_recon>0:
        #     rays_o_head = data['rays_o_head']
        #     rays_d_head = data['rays_d_head']
        #     HW_head = H if self.opt.dmtet else np.sqrt(rays_o_head.shape[1]).astype('int')
        #     H_head = H // 4
        #     W_head = W // 4
        #     bbox_head = [W//2-W//8, 0]
        #     outputs_head = self.model.render(rays_o_head, rays_d_head, mvp, HW_head, HW_head, 
        #                                      poses=poses, staged=True, perturb=False, light_d=light_d,
        #                                      ambient_ratio=ambient_ratio, shading=shading, bg_color=None, head_recon=True, bbox = bbox_head)
        #     pred_rgb_head = outputs_head['image'].reshape(B, H_head, W_head, 3)
        #     pred_depth_head = outputs_head['depth'].reshape(B, H_head, W_head)
        #     pred_mask_head = outputs_head['weights_sum'].reshape(B, H_head, W_head)
        #     if 'normal_image' in outputs_head:
        #         pred_normal_head = outputs_head['normal_image'].reshape(B, H_head, W_head, 3)
        #     return pred_rgb, pred_depth, pred_normal, pred_mask, pred_rgb_head, pred_depth_head, pred_normal_head, pred_mask_head, loss

        return pred_rgb, pred_depth, pred_normal, pred_mask, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']
        poses = data['poses']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        if self.opt.pose_path is not None:
            # dst_pose = data['hand_pose']
            # print('dst pose is not None')
            dst_pose = data['dst_pose']
        else:
            dst_pose = None
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, 
                                    poses=poses, staged=True, perturb=perturb, light_d=light_d, 
                                    ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color, dst_pose=dst_pose)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = None
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
        
        pred_rgb_head = pred_depth_head = pred_normal_head = None
        if self.head_recon>0:
            rays_o_head = data['rays_o_head']
            rays_d_head = data['rays_d_head']
            HW_head = H if self.opt.dmtet else np.sqrt(rays_o_head.shape[1]).astype('int')
            H_head = H // 4
            W_head = W // 4
            bbox_head = [W//2-W//8, 0]
            outputs_head = self.model.render(rays_o_head, rays_d_head, mvp, HW_head, HW_head, 
                                             poses=poses, staged=True, perturb=perturb, light_d=light_d, dst_pose=dst_pose, 
                                             ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color, head_recon=True, bbox=bbox_head)
            pred_rgb_head = outputs_head['image'].reshape(B, H_head, W_head, 3)
            pred_depth_head = outputs_head['depth'].reshape(B, H_head, W_head)
            if 'normal_image' in outputs_head:
                pred_normal_head = outputs_head['normal_image'].reshape(B, H_head, W_head, 3)

            return pred_rgb, pred_depth, pred_normal, pred_rgb_head, pred_depth_head, pred_normal_head

        return pred_rgb, pred_depth, pred_normal

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            # if self.workspace is not None and self.local_rank == 0:
                # self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                # self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)
                self.save_checkpoint(full=True, best=False)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            if self.opt.pose_path is None and not self.opt.eval_metrics and not self.opt.eval_supl:
                save_path = os.path.join(self.workspace, 'results')
            elif self.opt.eval_metrics:
                save_path = os.path.join(self.workspace, 'results_eval')
            elif self.opt.eval_supl:
                save_path = os.path.join(self.workspace, 'results_supl')
            else:
                if self.opt.handy_path is not None:
                    save_path = os.path.join(self.workspace, 'results_deform_handy')
                elif self.opt.mano_path is not None:
                    save_path = os.path.join(self.workspace, 'results_deform_mano')
                elif self.opt.smplx_path is not None:
                    save_path = os.path.join(self.workspace, 'results_deform_smplx')
                    

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []
            if self.head_recon>0:
                all_preds_head = []
                all_preds_depth_head = []
                all_preds_normal_head = []

        with torch.no_grad():

            # self.local_step = 0

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if self.head_recon>0:
                        preds, preds_depth, preds_normal, preds_head, preds_depth_head, preds_normal_head = self.test_step(data)
                    else:
                        preds, preds_depth, preds_normal = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                mask = pred_depth>1e-6
                # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                # pred_depth = (pred_depth * 255).astype(np.uint8)
                _pred_depth = pred_depth[mask]     
                _pred_depth = (_pred_depth - _pred_depth.min()) / (_pred_depth.max() - _pred_depth.min() + 1e-6)
                
                pred_depth[mask] = _pred_depth + 1/255
                pred_depth = (pred_depth * 255).astype(np.uint8)
                
                if preds_normal is not None:
                    pred_normal = preds_normal[0].detach().cpu().numpy()
                    pred_normal[pred_normal==0.5] = 1
                    pred_normal = (pred_normal * 255).astype(np.uint8)

                if self.head_recon>0:
                    pred_head = preds_head[0].detach().cpu().numpy()
                    pred_head = (pred_head * 255).astype(np.uint8)
                    pred_depth_head = preds_depth_head[0].detach().cpu().numpy()
                    # pred_depth_head = (pred_depth_head - pred_depth_head.min()) / (pred_depth_head.max() - pred_depth_head.min() + 1e-6)
                    pred_depth_head = (pred_depth_head * 255).astype(np.uint8)
                    if preds_normal_head is not None:
                        pred_normal_head = preds_normal_head[0].detach().cpu().numpy()
                        pred_normal_head = (pred_normal_head * 255).astype(np.uint8)
                
                if self.opt.eval_metrics:
                    dst_path = os.path.join(save_path, f'aeval_{i:04d}_rgb.png')
                    __mask = mask.astype(np.float32) * 255.
                    __pred = pred[:]
                    __pred[~mask] = np.array([255.,255.,255.])
                    __pred = np.concatenate((__pred, __mask[...,None]), axis=-1)
                    cv2.imwrite(dst_path, cv2.cvtColor(__pred, cv2.COLOR_RGBA2BGRA))
                    __pred_normal = pred_normal[:]
                    __pred_normal[~mask] = np.array([255.,255.,255.])
                    __pred_normal = np.concatenate((__pred_normal, __mask[...,None]), axis=-1)
                    cv2.imwrite(os.path.join(save_path, f'aeval_{i:04d}_normal.png'), cv2.cvtColor(__pred_normal, cv2.COLOR_RGBA2BGRA))
                
                if self.opt.test and self.opt.pose_path and not write_video:
                    if i % 1 == 0:
                        dst_path = os.path.join(save_path, f'aeval_{i:04d}_rgb.png')
                        __mask = mask.astype(np.float32) * 255.
                        __pred = pred[:]
                        __pred[~mask] = np.array([255.,255.,255.])
                        __pred = np.concatenate((__pred, __mask[...,None]), axis=-1)
                        cv2.imwrite(dst_path, cv2.cvtColor(__pred, cv2.COLOR_RGBA2BGRA))
                        __pred_normal = pred_normal[:]
                        __pred_normal[~mask] = np.array([255.,255.,255.])
                        __pred_normal = np.concatenate((__pred_normal, __mask[...,None]), axis=-1)
                        cv2.imwrite(os.path.join(save_path, f'aeval_{i:04d}_normal.png'), cv2.cvtColor(__pred_normal, cv2.COLOR_RGBA2BGRA))
                
                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    if preds_normal is not None:
                        all_preds_normal.append(pred_normal)
                    if self.head_recon>0:
                        all_preds_head.append(pred_head)
                        all_preds_depth_head.append(pred_depth_head)
                        if preds_normal_head is not None:
                            all_preds_normal_head.append(pred_normal_head)
                    # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    if preds_normal is not None:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'), cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR))
                    if self.head_recon>0 and not self.opt.dmtet:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb_head.png'), cv2.cvtColor(pred_head, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth_head.png'), pred_depth_head)
                        if preds_normal_head is not None:
                            cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal_head.png'), cv2.cvtColor(pred_normal_head, cv2.COLOR_RGB2BGR))

                pbar.update(loader.batch_size)

        if write_video:
            # all_preds = np.stack(all_preds, axis=0)
            # all_preds_depth = np.stack(all_preds_depth, axis=0)
            # all_preds_normal = np.stack(all_preds_normal, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            if preds_normal is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8, macro_block_size=1)
            if self.head_recon>0 and not self.opt.dmtet:
                # all_preds_head = np.stack(all_preds_head, axis=0)
                # all_preds_depth_head = np.stack(all_preds_depth_head, axis=0)
                imageio.mimwrite(os.path.join(save_path, f'{name}_rgb_head.mp4'), all_preds_head, fps=25, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, f'{name}_depth_head.mp4'), all_preds_depth_head, fps=25, quality=8, macro_block_size=1)
                if preds_normal_head is not None:
                    imageio.mimwrite(os.path.join(save_path, f'{name}_normal_head.mp4'), all_preds_normal_head, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs


    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        mvp = torch.from_numpy(mvp).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:

            # update grid every 16 steps
            if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                    save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, loss, loss_dict = self.train_step(data, save_guidance_path=save_guidance_path)
                pred_rgb = (pred_rgbs[0].permute(1,2,0).detach().cpu().numpy()*255.).astype(np.uint8)
                # cv2.imwrite('check_mesh/pred_rgb_zero123.png', pred_rgb)
            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
                pred_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()

            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    for k, v in loss_dict.items():
                        val = v.item()
                        if val > -1:
                            self.writer.add_scalar(f"train/{k}", val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # if self.head_recon>0:
                    #     preds, preds_depth, preds_normal, preds_mask, preds_head, preds_depth_head, preds_normal_head, preds_mask_head, loss = self.eval_step(data)
                    # else:
                    preds, preds_depth, preds_normal, preds_mask, loss = self.eval_step(data)

                

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    if preds_normal is not None:
                        preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_normal_list, preds)
                        preds_normal = torch.cat(preds_normal_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_normal = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_normal.png')
                    save_path_mask = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_mask.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    # pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_mask = preds_mask[0].detach().cpu().numpy()
                    pred_mask = (pred_mask * 255).astype(np.uint8)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)
                    cv2.imwrite(save_path_mask, pred_mask)

                    if preds_normal is not None:
                        pred_normal = preds_normal[0].detach().cpu().numpy()
                        # print(pred_normal.max(), pred_normal.min())
                        pred_normal = (pred_normal * 255).astype(np.uint8)
                        cv2.imwrite(save_path_normal, cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR))
                    
                    # if self.head_recon>0:
                    #     pred_head = preds_head[0].detach().cpu().numpy()
                    #     pred_head = (pred_head * 255).astype(np.uint8)
                    #     pred_depth_head = preds_depth_head[0].detach().cpu().numpy()
                    #     # pred_depth_head = (pred_depth_head - pred_depth_head.min()) / (pred_depth_head.max() - pred_depth_head.min() + 1e-6)
                    #     pred_depth_head = (pred_depth_head * 255).astype(np.uint8)

                    #     pred_mask_head = preds_mask_head[0].detach().cpu().numpy()
                    #     pred_mask_head = (pred_mask_head * 255).astype(np.uint8)

                    #     if preds_normal_head is not None:
                    #         pred_normal_head = preds_normal_head[0].detach().cpu().numpy()
                    #         pred_normal_head = (pred_normal_head * 255).astype(np.uint8)
                    #     save_head_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_head_rgb.png')
                    #     save_head_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_head_depth.png')
                    #     save_head_path_normal = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_head_normal.png')
                    #     save_head_path_mask = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_head_mask.png')
                    #     cv2.imwrite(save_head_path, cv2.cvtColor(pred_head, cv2.COLOR_RGB2BGR))
                    #     cv2.imwrite(save_head_path_depth, pred_depth_head)
                    #     cv2.imwrite(save_head_path_mask, pred_mask_head)
                    #     if preds_normal_head is not None:
                    #         cv2.imwrite(save_head_path_normal, cv2.cvtColor(pred_normal_head, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems
