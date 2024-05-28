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
import utils_opt
from ikhand.utils import mano_pose_16_to_21
from ikhand.math_np import convert
from mobhand.tools.joint_order import MANO2MPII, MPII2MANO
import ikhand.mesh_opt as mesh_opt
import iksolver.core as core


class MANOModel:
  def __init__(self, mano_path, camera, pts_reg=1.0, pose_reg=0.0, shape_reg=0.0, t_reg=0.0, unit=1):
      mano_model = utils_opt.load_official_mano_model(mano_path)
      self.unit = unit
      self.camera = camera
      self.kpts_mean = mano_model['kpts_mean'] * unit
      self.kpts_std = mano_model['kpts_std'] * unit
      self.parents = mano_model['parents']
      self.pts_reg = pts_reg
      self.pose_reg = pose_reg
      self.shape_reg = shape_reg
      self.t_reg = t_reg
      self.param_dim = {'pose': 63, 'shape': 10, 't': 3}
      self.weight = np.concatenate(
          [
              np.ones(21 * 2) * self.pts_reg,
              np.ones(self.param_dim['pose']) * self.pose_reg,
              np.ones(self.param_dim['shape']) * self.shape_reg,
              np.ones(3) * self.t_reg,
          ]
      )
      self.kinematics = core.Kinematics(
          self.kpts_mean, self.kpts_std, self.parents, 10, False, np.float32
      )
      self.k = np.zeros([21, 3])
      self.z = np.zeros([21, 1])
      self.k2 = np.zeros([21, 2])
      self.dk = np.zeros([21, 3, sum(self.param_dim.values())])

  def compose_params(self, params):
      params = np.concatenate([params[k].ravel() for k in self.param_dim])
      return params

  def decompose_params(self, params, mod=np):
      n = 0
      ret = {}
      for k, l in self.param_dim.items():
          ret[k] = params[n:n+l]
          n += l
      ret['pose'] = mod.array(mod.reshape(ret['pose'], [21, 3]))
      ret['shape'] = mod.array(mod.reshape(ret['shape'], [10]))
      ret['t'] = mod.array(mod.reshape(ret['t'], [1, 3]))
      return ret

  def forward_kinematics(self, params):
      self.k, self.dk = self.kinematics.forward(params, return_k=True)

  def projecttion(self):
      self.z = self.k[:, 2:3].copy()
      self.k2 = np.einsum('hw, jw -> jh', self.camera, self.k/self.z)[:, :2]
      return self.k2

  def residual(self, params):
      self.forward_kinematics(params)
      self.projecttion()
      output = np.concatenate([self.k2.ravel(), params])
      residual = self.weight * (output - self.target)
      return residual

  def jacobian(self):
      # for 2D projection
      drdk = np.zeros([42, 63])
      for i in range(21):
          drdk[2*i:2*(i+1), 3*i:3*(i+1)] = np.array([[self.weight[2*i] * self.camera[0, 0] / self.k[i, 2], 0, - self.weight[2*i] * self.camera[0, 0] * self.k[i, 0] / self.k[i, 2]**2],
                                                     [0, self.weight[2*i+1] * self.camera[1, 1] / self.k[i, 2], - self.weight[2*i+1] * self.camera[1, 1] * self.k[i, 1] / self.k[i, 2]**2]
                                                     ])
      jacobian_proj = np.matmul(drdk, self.dk.reshape(21*3, -1))

      # for param align
      jacobian_param = np.diag(self.weight[42:])

      # combine tow parts
      jacobian = np.concatenate([jacobian_proj, jacobian_param], 0)

      return jacobian

  def initialize(self, tgt_kpts, init_params):
      if init_params is None:
          init_params = {
              'pose': np.ones(21*3) * 1e-4,
              'shape': np.zeros(10),
              't': np.ones(3)
          }
      params = self.compose_params(init_params)
      self.target = np.concatenate([tgt_kpts.ravel(), params.copy()])
      return params


class LMSolver:
  def __init__(self, damp_factor=1e-2, max_step=10, early_stop=1e-5):
    self.max_step = max_step
    self.early_stop = early_stop
    self.damp_factor = damp_factor
    self.param_dim = {'pose': 63, 'shape': 10, 't': 3}

  def solve(self, params, model, verbose):
    last_mse = None
    damp_factor = self.damp_factor
    step = 0
    while step < self.max_step:

      residual = model.residual(params)
      mse = np.mean(np.square(residual))

      if last_mse is not None:
        if last_mse < mse:
          damp_factor *= 2
        else:
          damp_factor /= 2

          if last_mse - mse < self.early_stop:
            break

      j = model.jacobian()
      jtj = np.matmul(j.T, j)
      jtj = jtj + damp_factor * np.eye(jtj.shape[0])
      delta = np.matmul(np.matmul(np.linalg.inv(jtj), j.T), residual)
      params -= delta

      step += 1
      last_mse = mse

      if verbose:
        print(f'Step {step} | MSE {mse:.4e}')

    return params


if __name__ == '__main__':

    ik_file = '/Users/chenxingyu/Datasets/hand_test/ik/wrist_test/IMG_5108.MOV/ikpose_conv_oriz.json'
    save_dir = os.path.join( os.path.dirname(ik_file), ik_file.split('/')[-1].split('.')[0] )
    os.makedirs(save_dir, exist_ok=True)
    # K
    f = 1493
    c = 512
    cam_mat = np.array([
        [f, 0, c],
        [0, f, c],
        [0, 0, 1]
    ], np.float32)

    # Mano
    mano_layer = ManoLayer(
        rot_mode='axisang',
        use_pca=False,
        side='right',
        center_idx=0,
        mano_assets_root=os.path.join(os.path.dirname(__file__), '../template'),
        flat_hand_mean=True,
    )
    model = MANOModel('/Users/chenxingyu/Tools/mano_v1_2/models/MANO_RIGHT.pkl', cam_mat,
                        pts_reg=1e-3, pose_reg=1e-4, shape_reg=0.0, t_reg=0.0, unit=1)
    mano = mesh_opt.MeshModel('/Users/chenxingyu/Tools/mano_v1_2/models/MANO_RIGHT.pkl')

    # solver
    solver = LMSolver()
    solved_params = []

    # read data
    pose_dict_test = vc.load(ik_file)

    for i, sample in vc.progress_bar( enumerate( pose_dict_test.items() )):
        # if i<950:
        #     continue
        # read img
        image_path = sample[0]
        image_name = image_path.split('/')[-1]
        img = cv2.imread(image_path)

        # read
        verts = np.array( sample[1]['mano_verts'] )
        theta = np.array( sample[1]['theta'] )
        beta = np.array( sample[1]['beta'] )
        scale = np.array( sample[1]['scale'] )
        rel_scale = np.array( sample[1]['rel_scale'] )
        camera_r = np.array( sample[1]['camera_r'] )
        camera_t = np.array( sample[1]['camera_t'] )
        if camera_t[0, 2] < 0:
            camera_t[0, 2] *= -1
        else:
            camera_t[0, 1] *= -1

        # official mano inference
        mano_results = mano_layer(torch.from_numpy(theta).view(1, -1).float(), torch.from_numpy(beta).view(1, -1).float())
        verts_mano = mano_results.verts[0].cpu().numpy() * scale / rel_scale
        joints_mano = mano_results.joints[0].cpu().numpy() * scale / rel_scale

        # init opt param
        abs_theta = mano_pose_16_to_21(theta)
        abs_theta = convert(abs_theta, 'axangle', 'rotmat')
        z_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        abs_theta[0] = np.matmul(z_rot, abs_theta[0])
        abs_theta = vc.math_np.rotmat_rel_to_abs(abs_theta, skeletons.MANOHand.parents)
        abs_theta = convert(abs_theta, 'rotmat', 'axangle')
        tgt = np.zeros([21, 2])
        tgt[0] = np.array(sample[1]['2d'])[0]
        tgt[1] = (np.array(sample[1]['2d'])[0] + np.array(sample[1]['2d'])[1])/2
        tgt[2:] = np.array(sample[1]['2d'])[1:]
        tgt = tgt[MPII2MANO]
        init_param = model.initialize(tgt, {'pose': abs_theta, 'shape': beta, 't': camera_t})

        # solve
        # model.forward_kinematics(init_param)
        # model.projecttion()
        params = solver.solve(init_param, model, verbose=False)
        solved_params.append(params)
        params = model.decompose_params(params, mod=np)
        rotmat = convert(np.reshape(params['pose'], [21, 3]), 'axangle', 'rotmat')
        kpts_opt, verts_opt = mano.set_params(rotmat, params['shape'])
        verts_opt -= kpts_opt[0]
        # verts_opt[:, :2] *= -1
        verts_opt += params['t']
        # kpts_opt -= kpts_opt[0]

        # global
        verts_global = verts_mano.copy()
        verts_global[:, 0:2] *= -1
        verts_global += camera_t

        # joint
        joints_global = joints_mano.copy()
        joints_global[:, 0:2] *= -1
        joints_global += camera_t

        # uv
        uv = sample[1]['2d']
        uv_plot = vc.render_bones_from_uv(np.flip(uv, axis=-1).copy(), img.copy(), skeletons.KWAIHand)
        uv_plot = cv2.resize(uv_plot, (uv_plot.shape[1]//4, uv_plot.shape[0]//4))[..., :3]

        # plt
        plt_plot = img.copy()
        plt_plot = draw_aligned_mesh_plt(plt_plot.copy(), cam_mat, verts_global, mano_layer.th_faces, lw=2)

        # 3D pose
        joint2uv = np.matmul(cam_mat, joints_global.T).T
        joint2uv = (joint2uv / joint2uv[:, 2:3])[:, :2].astype(np.int32)
        plt_plot = vc.render_bones_from_uv(np.flip(joint2uv, axis=-1).copy(), plt_plot.copy(), skeletons.MPIIHand)
        plt_plot = cv2.resize(plt_plot, (plt_plot.shape[1]//4, plt_plot.shape[0]//4))[..., :3]

        model_plot = img.copy()
        model_plot = draw_aligned_mesh_plt(model_plot.copy(), cam_mat, verts_opt, mano_layer.th_faces, lw=2)
        model_plot = vc.render_bones_from_uv(np.flip(model.k2[MANO2MPII], axis=-1).copy(), model_plot.copy(), skeletons.MPIIHand)
        model_plot = cv2.resize(model_plot, (model_plot.shape[1]//4, model_plot.shape[0]//4))[..., :3]

        # display
        display = np.concatenate([uv_plot, plt_plot, model_plot], 1)
        # cv2.imshow('test', display)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, image_name), display)
