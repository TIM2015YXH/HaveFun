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
import jax
import jax.numpy as jnp
import utils_opt
from ikhand.utils import mano_pose_16_to_21
from ikhand.math_np import convert
from mobhand.tools.joint_order import MANO2MPII
from ikhand.optimizer import LMSolver
import ikhand.mesh_opt as mesh_opt


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
      self.jacobian_fn = jax.jacfwd(self.residual)
      self.weight = np.concatenate(
          [
              np.ones(21 * 2) * self.pts_reg,
              np.ones(self.param_dim['pose']) * self.pose_reg,
              np.ones(self.param_dim['shape']) * self.shape_reg,
              np.ones(3) * self.t_reg,
          ]
      )
      self.jacobian = jax.jacfwd(self.residual)

  def compose_params(self, params):
      params = jnp.concatenate([params[k].ravel() for k in self.param_dim])
      return params

  def decompose_params(self, params, mod=jnp):
      n = 0
      ret = {}
      for k, l in self.param_dim.items():
          ret[k] = params[n:n+l]
          n += l
      ret['pose'] = mod.array(mod.reshape(ret['pose'], [21, 3]))
      ret['shape'] = mod.array(mod.reshape(ret['shape'], [10]))
      ret['t'] = mod.array(mod.reshape(ret['t'], [1, 3]))
      return ret

  def kpts_to_bones(self, kpts):
      bones = []
      for c, p in enumerate(self.parents):
          if p is None:
              bones.append(kpts[c])
          else:
              bones.append(kpts[c] - kpts[p])
      bones = jnp.stack(bones)
      return bones

  def bones_to_kpts(self, bones):
      kpts = []
      for c, p in enumerate(self.parents):
          if p is None:
              kpts.append(bones[c])
          else:
              kpts.append(bones[c] + kpts[p])
      kpts = jnp.stack(kpts)
      return kpts

  def axangle_to_rotmat(self, axangle):
      theta = jnp.linalg.norm(axangle, axis=-1, keepdims=True)
      c = jnp.cos(theta)
      s = jnp.sin(theta)
      t = 1 - c
      x, y, z = jnp.split(axangle / theta, 3, axis=-1)
      rotmat = jnp.stack([
          t * x * x + c,     t * x * y - z * s, t * x * z + y * s,
          t * x * y + z * s, t * y * y + c,     t * y * z - x * s,
          t * x * z - y * s, t * y * z + x * s, t * z * z + c
      ], 1)
      rotmat = jnp.reshape(rotmat, [-1, 3, 3])
      return rotmat

  def forward_kinematics(self, params):
      params = self.decompose_params(params)
      abs_rotmat = self.axangle_to_rotmat(params['pose'])
      ref_kpts = \
          self.kpts_mean + jnp.einsum('c, cjd->jd', params['shape'], self.kpts_std)
      ref_bones = self.kpts_to_bones(ref_kpts)
      bones = jnp.einsum('jhw, jw -> jh', abs_rotmat, ref_bones)
      kpts = self.bones_to_kpts(bones)
      return kpts, params['t']

  def projecttion(self, kpts, t):
      kpts -= kpts[0]
      kpts_flip = kpts[:, 0:2] * -1
      kpts2d = jnp.concatenate([kpts_flip, kpts[:, 2:]], axis=1)
      kpts2d = jnp.matmul(self.camera, (kpts2d + t).T).T
      kpts2d = (kpts2d / kpts2d[:, 2:3])[MANO2MPII, :2]
      return kpts2d

  def residual(self, params):
      kpts, t = self.forward_kinematics(params)
      kpts2d = self.projecttion(kpts, t)
      # res_list = np.array([i for i, l in enumerate(skeletons.MPIIHand.labels) if 'T0' not in l])
      output = jnp.concatenate([kpts2d.ravel(), params])
      residual = self.weight * (self.target - output)
      return residual

  def initialize(self, tgt_kpts, init_params):
      if init_params is None:
          init_params = {
              'pose': np.ones(21*3) * 1e-4,
              'shape': np.zeros(10),
              't': np.ones(3)
          }
      params = jnp.array(self.compose_params(init_params))
      self.target = np.concatenate([tgt_kpts.ravel(), params.copy()])
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
        abs_theta = vc.math_np.rotmat_rel_to_abs(abs_theta, skeletons.MANOHand.parents)
        abs_theta = convert(abs_theta, 'rotmat', 'axangle')
        # tgt_list = [i for i, l in enumerate(skeletons.KWAIHand.labels) if '0' not in l]
        tgt = np.zeros([21, 2])
        tgt[0] = np.array(sample[1]['2d'])[0]
        tgt[1] = (np.array(sample[1]['2d'])[0] + np.array(sample[1]['2d'])[1])/2
        tgt[2:] = np.array(sample[1]['2d'])[1:]
        init_param = model.initialize(tgt, {'pose': abs_theta, 'shape': beta, 't': camera_t})

        # solve
        params = solver.solve(init_param, model, verbose=False)
        solved_params.append(params)
        params = model.decompose_params(params, mod=np)
        rotmat = convert(np.reshape(params['pose'], [21, 3]), 'axangle', 'rotmat')
        kpts_opt, verts_opt = mano.set_params(rotmat, params['shape'])
        verts_opt -= kpts_opt[0]
        verts_opt[:, :2] *= -1
        verts_opt += params['t']
        kpts_opt -= kpts_opt[0]
        kpts2d_opt = np.asarray(model.projecttion(kpts_opt, params['t']))

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
        model_plot = vc.render_bones_from_uv(np.flip(kpts2d_opt, axis=-1).copy(), model_plot.copy(), skeletons.MPIIHand)
        model_plot = cv2.resize(model_plot, (model_plot.shape[1]//4, model_plot.shape[0]//4))[..., :3]

        # display
        display = np.concatenate([uv_plot, plt_plot, model_plot], 1)
        # cv2.imshow('test', display)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, image_name), display)
