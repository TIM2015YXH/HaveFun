import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cv2
from manotorch.manolayer import ManoLayer
import vctoolkit as vc
from utils.draw3d import draw_aligned_mesh_plt
from ikhand import skeletons
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobhand.tools.joint_order import MANO2MPII
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, SoftPhongShader,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, TexturesUV
)
from ikhand.data import MANODatasetOriginal
from ikhand.utils import mano_pose_16_to_21
from ikhand.math_np import convert
import ikhand.math_pt as math_pt
import utils_opt


class ProjectionLoss(nn.Module):
    def __init__(self, f, c):
        super(ProjectionLoss, self).__init__()
        self.cameras = PerspectiveCameras(focal_length=f,
                                          principal_point=((c, c),),
                                          image_size=((1024, 1024),)
                                          )

    def forward(self, xyz, gt):
        xy = self.cameras.transform_points(xyz)[:, :2]
        xy = (xy * 0.5 + 0.5) * self.cameras.image_size

        loss = F.l1_loss(xy, gt, reduction='mean')
        return loss, xy


class SilhouetteLoss(nn.Module):
    def __init__(self, f, c, face):
        super(SilhouetteLoss, self).__init__()
        self.cameras = PerspectiveCameras(focal_length=f,
                                          principal_point=((c, c),),
                                          image_size=((1024, 1024),)
                                          )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=(1920//8, 1080//8),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        self.R = torch.tensor([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]]).float()
        self.face = face

    def compute_iou(self, pred, gt):
        area_pred = pred.sum()
        area_gt = gt.sum()
        if area_pred == area_gt == 0:
            return 0
        union_area = (pred + gt).clip(max=1)
        union_area = union_area.sum()
        inter_area = area_pred + area_gt - union_area
        IoU = inter_area / union_area

        return 1-IoU

    def forward(self, verts, gt):
        verts = torch.mm(self.R, verts.T).T
        mesh = Meshes(
            verts=[verts],
            faces=[self.face],
        )
        silhouette = self.silhouette_renderer(meshes_world=mesh)[0, :, :, 3]
        # loss = self.compute_iou(silhouette, gt)
        loss = F.l1_loss(silhouette, gt, reduction='mean')
        cv2.imwrite('test.png', (silhouette.detach().numpy()*255).astype(np.uint8))
        # cv2.waitKey(0)
        return loss, silhouette


class MANOModel(nn.Module):
  def __init__(self, f, c, pts_reg=1e-3, pose_reg=0.0, shape_reg=0.0, t_reg=0.0, sil_reg=1.0):
      super(MANOModel, self).__init__()
      # mano model
      mano_model = utils_opt.load_official_mano_model('/Users/chenxingyu/Tools/mano_v1_2/models/MANO_RIGHT.pkl')
      self.kpts_mean = torch.from_numpy(mano_model['kpts_mean']).float()
      self.kpts_std = torch.from_numpy(mano_model['kpts_std']).float()
      self.verts_mean = torch.from_numpy(mano_model['mesh']).float()
      self.shape_basis = torch.from_numpy(mano_model['shape_basis']).float()
      self.weights = torch.from_numpy(mano_model['weights']).float()
      self.parents = mano_model['parents']
      self.faces = mano_model['faces']
      self.j_regressor = torch.from_numpy(mano_model['keypoint_regressor']).float()
      self.rel_scale = 1.0
      # params
      self.theta = nn.Parameter(torch.zeros(21, 3))
      self.beta = nn.Parameter(torch.zeros(10))
      self.t = nn.Parameter(torch.zeros(3))
      self.t_sil = nn.Parameter(torch.zeros(3))
      # target
      self.uv_gt = torch.zeros(21, 2)
      self.theta_init = torch.zeros(21, 3)
      self.beta_init = torch.zeros(10)
      self.t_init = torch.zeros(3)
      # results
      self.verts = torch.zeros(778, 3)
      self.joints = torch.zeros(21, 3)
      self.uv_proj = torch.zeros(21, 2)
      # weight
      self.pts_reg = pts_reg
      self.pose_reg = pose_reg
      self.shape_reg = shape_reg
      self.t_reg = t_reg
      self.sil_reg = sil_reg
      # loss
      self.proj_loss = ProjectionLoss(f, c)
      self.sil_loss = SilhouetteLoss(f, c, torch.from_numpy(self.faces.astype(np.int32)))

  def axangle_to_rotmat(self, axangle):
      theta = torch.norm(axangle, dim=-1, keepdim=True)
      c = torch.cos(theta)
      s = torch.sin(theta)
      t = 1 - c
      x, y, z = torch.split(axangle / theta, 1, dim=-1)
      rotmat = torch.stack([
          t * x * x + c,     t * x * y - z * s, t * x * z + y * s,
          t * x * y + z * s, t * y * y + c,     t * y * z - x * s,
          t * x * z - y * s, t * y * z + x * s, t * z * z + c
      ], 1)
      rotmat = torch.reshape(rotmat, [-1, 3, 3])
      return rotmat

  def kpts_to_bones(self, kpts):
      bones = []
      for c, p in enumerate(self.parents):
          if p is None:
              bones.append(kpts[c])
          else:
              bones.append(kpts[c] - kpts[p])
      bones = torch.stack(bones)
      return bones

  def bones_to_kpts(self, bones):
      kpts = []
      for c, p in enumerate(self.parents):
          if p is None:
              kpts.append(bones[c])
          else:
              kpts.append(bones[c] + kpts[p])
      kpts = torch.stack(kpts)
      return kpts

  def forward(self):
      abs_rotmat = self.axangle_to_rotmat(self.theta)
      verts = self.verts_mean + torch.einsum('c, vdc -> vd', self.beta, self.shape_basis)
      ref_kpts = torch.einsum('vd, jv -> jd', verts, self.j_regressor)
      bones = self.kpts_to_bones(ref_kpts)
      bones = torch.einsum('jhw, jw -> jh', abs_rotmat, bones)
      kpts = self.bones_to_kpts(bones)
      j = kpts - torch.einsum('jhw, jw -> jh', abs_rotmat, ref_kpts)
      g = torch.cat([abs_rotmat, j.unsqueeze(-1)], -1)
      verts = torch.cat([verts, torch.ones(verts.shape[:-1] + (1,))], -1)
      verts = torch.einsum(
          'vj, jvd -> vd', self.weights, torch.einsum('jhw, vw -> jvh', g, verts)
      )

      self.joints = (kpts[MANO2MPII] - kpts[0]) / self.rel_scale
      self.joints[:, 0:2] *= -1
      self.joints += self.t
      self.verts = (verts - kpts[0]) / self.rel_scale
      self.verts[:, 0:2] *= -1
      self.verts += self.t
      self.verts_sil = self.verts + self.t_sil

  def initialize(self, theta, beta, t, uv_gt, rel_scale, mask):
      self.theta.data = theta.clone()
      self.beta.data = beta.clone()
      self.t.data = t.clone()
      self.t_sil.data = torch.zeros(3)
      self.theta_init = theta.clone()
      self.beta_init = beta.clone()
      self.t_init = t.clone()
      self.verts = torch.zeros(778, 3)
      self.joints = torch.zeros(21, 3)
      self.uv_gt = uv_gt
      self.mask_gt = mask
      self.uv_proj = torch.zeros(21, 2)
      self.rel_scale = rel_scale

  def loss(self):
      loss_pts, self.uv_proj = self.proj_loss(self.joints, self.uv_gt)
      loss_pts *= self.pts_reg

      loss_sil, self.sil_proj = self.sil_loss(self.verts_sil, self.mask_gt)
      loss_sil *= self.sil_reg

      loss_theta = F.l1_loss(self.theta, self.theta_init) * self.pose_reg
      loss_beta = torch.norm(self.beta) * self.shape_reg
      # loss_beta = F.l1_loss(self.beta, self.beta_init) * self.shape_reg
      loss_t = F.l1_loss(self.t, self.t_init) * self.t_reg
      # loss_beta = torch.norm(self.beta) * self.shape_reg

      loss = loss_pts + loss_theta + loss_beta + loss_t + loss_sil
      return loss

if __name__ == '__main__':

    ik_file = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikpose_conv_oriz.json'
    save_dir = os.path.join( os.path.dirname(ik_file), ik_file.split('/')[-1].split('.')[0] )
    os.makedirs(save_dir, exist_ok=True)
    # camera
    f = 1493
    c = 512
    cam_mat = np.array([
        [f, 0, c],
        [0, f, c],
        [0, 0, 1]
    ], np.float32)

    # Mano
    model = MANOModel(f, c, pts_reg=1e-3, pose_reg=1e-4, shape_reg=1e-2, t_reg=0.0, sil_reg=1e-1)
    print(sum([m.numel() for m in model.parameters() if m.requires_grad]))

    # opt
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(
    #                             [{'params': model.theta, 'lr': 0.1},
    #                              {'params': model.beta, 'lr': 0.0001},
    #                              {'params': model.t, 'lr': 0.1},], 0.001)


    # optimizer = torch.optim.SGD([{'params': model.theta, 'lr': 0.001},
    #                              {'params': model.beta, 'lr': 0.0001},
    #                              {'params': model.t, 'lr': 0.001},], lr=0.1, momentum=0.9)

    # read data
    pose_dict_test = vc.load(ik_file)

    for i, sample in vc.progress_bar( enumerate( pose_dict_test.items() )):
        # if i<950:
        #     continue
        # read img
        image_path = sample[0]
        image_name = image_path.split('/')[-1]
        img = cv2.imread(image_path)
        mask_path = image_path.replace(f'/{image_name}', f'_mask/{image_name}').replace('jpg', 'png')
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (1080//8, 1920//8))[..., 0]

        # read
        verts = np.array( sample[1]['mano_verts'] )
        theta = np.array( sample[1]['theta'] )
        beta = np.array( sample[1]['beta'] )
        scale = np.array( sample[1]['scale'] )
        rel_scale = np.array( sample[1]['rel_scale'] )
        camera_r = np.array( sample[1]['camera_r'] )
        camera_t = np.array( sample[1]['camera_t'] )
        uv_gt = np.array( sample[1]['2d'] )

        if camera_t[0, 2] < 0:
            camera_t[0, 2] *= -1
        else:
            camera_t[0, 1] *= -1
        # pose convert
        theta = mano_pose_16_to_21(theta)
        rot = convert(theta, 'axangle', 'rotmat')
        abs_rot = vc.math_np.rotmat_rel_to_abs(rot, skeletons.MANOHand.parents)
        abs_theta = convert(abs_rot, 'rotmat', 'axangle')
        # official mano inference
        abs_theta = torch.from_numpy(abs_theta).float()
        beta = torch.from_numpy(beta).float()
        t = torch.from_numpy(camera_t).float()
        uv_gt = np.zeros([21, 2])
        uv_gt[0] = np.array(sample[1]['2d'])[0]
        uv_gt[1] = (np.array(sample[1]['2d'])[0] + np.array(sample[1]['2d'])[1])/2
        uv_gt[2:] = np.array(sample[1]['2d'])[1:]
        uv_gt = torch.from_numpy(uv_gt).view(-1, 2).float()
        rel_scale = torch.from_numpy(rel_scale).float()
        mask = torch.from_numpy(mask).float() / 255.0
        model.initialize(abs_theta, beta, t, uv_gt, rel_scale, mask)
        model.forward()
        joints = model.joints.detach().numpy()
        verts = model.verts.detach().numpy()

        # uv
        uv = sample[1]['2d']
        uv_plot = vc.render_bones_from_uv(np.flip(uv_gt.numpy(), axis=-1).copy(), img.copy(), skeletons.MPIIHand)
        uv_plot = cv2.resize(uv_plot, (uv_plot.shape[1]//4, uv_plot.shape[0]//4))[..., :3]
        # plt
        plt_plot = img.copy()
        plt_plot = draw_aligned_mesh_plt(plt_plot.copy(), cam_mat, verts, model.faces, lw=2)
        # 3D pose
        joint2uv = np.matmul(cam_mat, joints.T).T
        joint2uv = (joint2uv / joint2uv[:, 2:3])[:, :2].astype(np.int32)
        plt_plot = vc.render_bones_from_uv(np.flip(joint2uv, axis=-1).copy(), plt_plot.copy(), skeletons.MPIIHand)
        plt_plot = cv2.resize(plt_plot, (plt_plot.shape[1]//4, plt_plot.shape[0]//4))[..., :3]

        # opt
        loss = 1.
        attemp = 0
        while loss > 0.01:
            optimizer.zero_grad()
            model.forward()
            loss = model.loss()
            print(attemp, loss, model.t_sil.data)
            loss.backward()
            optimizer.step()
            attemp += 1

        model_plot = img.copy()
        model_plot = draw_aligned_mesh_plt(model_plot.copy(), cam_mat, model.verts.detach().numpy(), model.faces, lw=2)
        model_plot = vc.render_bones_from_uv(np.flip(model.uv_proj.detach().numpy().astype(np.int32), axis=-1).copy(), model_plot.copy(), skeletons.MPIIHand)
        model_plot = cv2.resize(model_plot, (model_plot.shape[1]//4, model_plot.shape[0]//4))[..., :3]

        # display
        display = np.concatenate([uv_plot, plt_plot, model_plot], 1)
        # cv2.imshow('test', display)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, image_name), display)
