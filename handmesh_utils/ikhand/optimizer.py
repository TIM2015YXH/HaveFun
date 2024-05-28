import utils_opt
import jax
import jax.numpy as jnp
import numpy as np
import vctoolkit as vc
from settings import GLOBAL


class MANOModel:
  def __init__(self, mano_path, pose_reg=0.0, shape_reg=0.0, scale_reg=0.0, unit=1):
    mano_model = utils_opt.load_official_mano_model(mano_path)
    self.unit = unit
    self.kpts_mean = mano_model['kpts_mean'] * unit
    self.kpts_std = mano_model['kpts_std'] * unit
    self.parents = mano_model['parents']
    self.pose_reg = pose_reg
    self.shape_reg = shape_reg
    self.scale_reg = scale_reg
    self.param_dim = {'pose': 63, 'shape': 10, 'scale': 1}
    self.jacobian_fn = jax.jacfwd(self.residual)
    self.weight = np.concatenate(
      [
        np.ones(21 * 3),
        np.ones(self.param_dim['pose']) * np.sqrt(self.pose_reg),
        np.ones(self.param_dim['shape']) * np.sqrt(self.shape_reg),
        np.ones(1) * np.sqrt(self.scale_reg),
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
    ret['scale'] = mod.array(mod.reshape(ret['scale'], [1]))
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
    kpts = self.bones_to_kpts(bones) * params['scale']
    return kpts

  def residual(self, params):
    kpts = self.forward_kinematics(params)
    output = jnp.concatenate([kpts.ravel(), params])
    residual = self.weight * (self.target - output)
    return residual

  def initialize(self, tgt_kpts, init_params=None):
    if init_params is None:
      init_params = {
        'pose': np.ones(63) * 1e-4,
        'shape': np.zeros(10),
        'scale': np.ones(1)
      }
    params = jnp.array(self.compose_params(init_params))
    self.target = np.concatenate([tgt_kpts.ravel() * self.unit, params.copy()])
    return params


class LMSolver:
  def __init__(self, damp_factor=1e-2, max_step=10, early_stop=1e-5):
    self.max_step = max_step
    self.early_stop = early_stop
    self.damp_factor = damp_factor
    self.param_dim = {'pose': 63, 'shape': 10, 'scale': 1}

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

      j = model.jacobian(params)
      jtj = jnp.matmul(j.T, j)
      jtj = jtj + damp_factor * jnp.eye(jtj.shape[0])
      delta = jnp.matmul(jnp.matmul(jnp.linalg.inv(jtj), j.T), residual)
      params -= delta

      step += 1
      last_mse = mse

      if verbose:
        print(f'Step {step} | MSE {mse:.4e}')

    return params


if __name__ == '__main__':
  from vctoolkit import math_np
  import mesh_opt
  import os
  import skeletons as sk

  np.random.seed(1214)
  mano = mesh_opt.MeshModel(GLOBAL.paths['mano_left'])

  model = MANOModel(
    GLOBAL.paths['mano_left'], pose_reg=1e-4, shape_reg=0.0, scale_reg=0.0,
    unit=10
  )
  solver = LMSolver()

  model_name = 'iknet_v14_precise'
  data = vc.load_pkl(f'output/{model_name}/fist/fist.pkl')

  save_dir = os.path.join('output', model_name, 'fist', 'optimize')
  os.makedirs(save_dir, exist_ok=True)

  solved_params = []
  for idx in vc.progress_bar(range(data['gt_kpts'].shape[0])):
    init_params = {
      'pose': math_np.convert(np.reshape(data['ik']['pose'][idx], [21, 6]), 'rot6d', 'axangle'),
      'shape': data['ik']['shape'][idx],
      'scale': data['ik']['scale'][idx],
    }
    init_params = model.initialize(data['gt_kpts'][idx], init_params)
    params = solver.solve(init_params, model, verbose=False)
    solved_params.append(params)

  kpts_err = []
  verts_err = []
  for idx, params in enumerate(solved_params):
    params = model.decompose_params(params, mod=np)
    rotmat = \
      math_np.convert(np.reshape(params['pose'], [21, 3]), 'axangle', 'rotmat')
    kpts, verts = mano.set_params(rotmat, params['shape'], scale=params['scale'])

    kpts_err.append(
      np.mean(np.linalg.norm(data['gt_kpts'][idx] - kpts, axis=-1)) * 1000
    )
    verts_err.append(
      np.mean(np.linalg.norm(data['gt_verts'][idx] - verts, axis=-1)) * 1000
    )

    vc.save(
      os.path.join(save_dir, f'{idx}_gt_mesh.obj'),
      (data['gt_verts'][idx], mano.faces)
    )
    vc.save(
      os.path.join(save_dir, f'{idx}_pred_mesh.obj'),
      (verts, mano.faces)
    )

    vc.joints_to_mesh(
      data['gt_kpts'][idx], sk.MANOHand,
      save_path=os.path.join(save_dir, f'{idx}_gt_kpts.obj'),
    )
    vc.joints_to_mesh(
      kpts, sk.MANOHand,
      save_path=os.path.join(save_dir, f'{idx}_pred_kpts.obj'),
    )

  print(
    f'Keypoint error = {np.mean(kpts_err):.2f}mm | '
    f'Vertex error = {np.mean(verts_err):.2f}mm'
  )
