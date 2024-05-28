import vctoolkit as vc
import numpy as np
import os
import config as cfg
import math_np
import skeletons as sk
import utils
import data
import matplotlib.pyplot as plt
import mesh


def examine_selected_samples():
  mano = mesh.MeshModel(cfg.MANO_PATH)
  tar_dir = 'output/yajiao_xyz/selected'
  os.makedirs(tar_dir, exist_ok=True)
  for sample in cfg.SELECTED_FRAMES:
    video, frame_idx = sample.split(os.sep)[-2:]
    frame_idx = os.path.splitext(frame_idx)[0]

    data = vc.load(os.path.join('output', 'yajiao_xyz', f'{video}.pkl'))
    abs_rot6d = np.reshape(data[sample]['pose'], [-1, 6])
    shape = data[sample]['shape']
    scale = data[sample]['scale']

    abs_rotmat = math_np.convert(abs_rot6d, 'rot6d', 'rotmat')
    _, v = mano.set_params(abs_rotmat, shape)
    v = v * scale * cfg.MANO_SCALE
    v, f = utils.left_right_flip(v, mano.faces)
    vc.save_obj(os.path.join(tar_dir, f'{video}_{frame_idx}_mesh.obj'), v, f)

    # if we run it on the laptop we do not have the image
    if os.path.exists(sample):
      img = vc.load(sample)
      vc.save(os.path.join(tar_dir, f'{video}_{frame_idx}_image.jpg'), img)

    v, f = vc.joints_to_mesh(data[sample]['xyz'], sk.KWAIHand.parents)
    v, f = utils.left_right_flip(v, f)
    vc.save_obj(os.path.join(tar_dir, f'{video}_{frame_idx}_xyz.obj'), v, f)


def examine_dataloader_distribution():
  dataset = data.MANODatasetINTRPL(p_power=2, mix_fingers=True)
  n = 10000
  np.random.seed(1019)
  cache = []
  for i in range(n):
    quat = dataset[0]['rel_quat']
    degrees = np.rad2deg(np.arccos(quat[:, 0]) * 2)
    cache.append(degrees)
  cache = np.stack(cache)

  n_rows = 5
  n_cols = 3
  plt.figure(figsize=(n_cols * 5, n_rows * 2))
  for i, f in enumerate('TIMRL'):
    for j, k in enumerate('123'):
      plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
      plt.hist(cache[:, sk.MANOHand.labels.index(f + k)], bins=100)
      plt.title(f + k)
      plt.xlabel('degree')
      plt.ylabel('N')
  plt.tight_layout()
  plt.show()


def examine_raw_data_distribution():
  axangle = utils.load_official_mano_model(cfg.MANO_PATH)['rel_axangle']
  quat = math_np.convert(axangle, 'axangle', 'quat')
  pose = math_np.convert(quat, 'quat', 'axangle')

  angles = np.rad2deg(np.linalg.norm(pose, axis=-1))

  n_rows = 5
  n_cols = 3
  plt.figure(figsize=(n_cols * 5, n_rows * 2))
  for i, f in enumerate('TIMRL'):
    for j, k in enumerate('123'):
      plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
      plt.hist(angles[:, sk.MANOHand.labels.index(f + k)], bins=100)
      plt.title(f + k)
      plt.xlabel('degree')
      plt.ylabel('N')
  plt.tight_layout()
  plt.show()


def examine_mano_model_loading():
  np.random.seed(1019)
  tar_dir = os.path.join(cfg.CODE_TEST_DIR, 'mano_model')
  os.makedirs(tar_dir, exist_ok=True)
  mano = utils.load_official_mano_model(cfg.MANO_PATH)
  poses = mano['rel_axangle']
  ref_keypoints = mano['keypoints_mean']
  ref_bones = math_np.keypoints_to_bones(ref_keypoints, sk.MANOHand.parents)

  mesh = math_np.MeshModel(cfg.MANO_PATH)

  n = 10
  samples = poses[np.random.choice(poses.shape[0], n)]
  for i, rel_axangle in enumerate(samples):
    rel_rotmat = math_np.convert(rel_axangle, 'axangle', 'rotmat')
    abs_rotmat = math_np.rotmat_rel_to_abs(rel_rotmat, sk.MANOHand.parents)
    keypoints, _ = math_np.forward_kinematics(ref_bones, abs_rotmat, sk.MANOHand.parents)
    v, f = vc.joints_to_mesh(keypoints, sk.MANOHand.parents)
    vc.save_obj(os.path.join(tar_dir, f'keypoints_{i}.obj'), v, f)
    _, v = mesh.set_params(abs_rotmat)
    vc.save_obj(os.path.join(tar_dir, f'mesh_{i}.obj'), v, mesh.faces)


def examine_mano_dataloader():
  tar_dir = os.path.join('output', 'mano_dataloader')
  os.makedirs(tar_dir, exist_ok=True)

  settings = {
    'shape_std': 1,
    'scale_std': 0.1,
    'conf_mean': 0.9,
    'conf_std': 0.0,
    'noise_std': 0.1,
    'p_power': 2,
    'batch_size': 8,
    'mix_fingers': False,
  }
  mano = mesh.MeshModel(cfg.MANO_PATH)
  dataset = data.MANODatasetINTRP(settings)
  np.random.seed(1019)
  pack = dataset[0]

  for i in range(settings['batch_size']):
    abs_rotmat = \
      math_np.convert(np.reshape(pack['pose'][i], [-1, 6]), 'rot6d', 'rotmat')

    mesh_keypoints, v = mano.set_params(abs_rotmat, pack['shape'][i])
    vc.save_obj(
      os.path.join(tar_dir, f'mesh_{i}.obj'),
      v * pack['scale'][i] * cfg.MANO_SCALE, mano.faces
    )

    v, f = vc.joints_to_mesh(mesh_keypoints, sk.MANOHand.parents)
    vc.save_obj(os.path.join(tar_dir, f'mesh_kpts_{i}.obj'), v, f)

    v, f = vc.joints_to_mesh(
      np.reshape(pack['inputs'][i], [-1, 4])[:, :3], sk.MANOHand.parents
    )
    vc.save_obj(os.path.join(tar_dir, f'input_kpts_{i}.obj'), v, f)

    v, f = vc.joints_to_mesh(pack['keypoints'][i], sk.MANOHand.parents)
    vc.save_obj(os.path.join(tar_dir, f'gt_kpts_{i}.obj'), v, f)


def examine_mano_all_vert_dataloader():
  tar_dir = os.path.join('output', 'mano_all_vert_dataloader')
  os.makedirs(tar_dir, exist_ok=True)

  settings = {
    'shape_std': 1,
    'scale_std': 0.1,
    'conf_mean': 0.9,
    'conf_std': 0.0,
    'noise_std': 0.1,
    'p_power': 2,
    'batch_size': 8,
    'mix_fingers': False,
  }
  mano = mesh.MeshModel(cfg.MANO_PATH)
  dataset = data.MANODatasetVertWrapper(data.MANODatasetINTRP(settings))
  np.random.seed(1019)
  pack = dataset[0]

  for i in range(settings['batch_size']):
    abs_rotmat = \
      math_np.convert(np.reshape(pack['pose'][i], [-1, 6]), 'rot6d', 'rotmat')

    _, v = mano.set_params(abs_rotmat, pack['shape'][i])
    vc.save_obj(
      os.path.join(tar_dir, f'mesh_gt_{i}.obj'),
      v * pack['scale'][i] * cfg.MANO_SCALE, mano.faces
    )

    v = np.reshape(pack['inputs'][i], [-1, 3])
    vc.save_obj(os.path.join(tar_dir, f'mesh_input_{i}.obj'), v, mano.faces)


def test_axangle_to_rotmat():
  N = 1000
  np.random.seed(1019)
  for _ in range(N):
    axis = np.random.uniform(size=(3))
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(np.pi)
    axangle = axis * angle
    rotmat = math_np.convert(axangle, 'axangle', 'rotmat')
    axangle_ = math_np.convert(rotmat, 'rotmat', 'axangle')
    assert(np.allclose(axangle, axangle_))
  print('Good.')


def test_rotmat_rel_abs():
  N = 1000
  np.random.seed(1019)
  for _ in range(N):
    rel_rotmat = np.stack([math_np.random_rotation() for _ in range(sk.MANOHand.n_joints)])
    abs_rotmat = math_np.rotmat_rel_to_abs(rel_rotmat, sk.MANOHand.parents)
    rel_rotmat_ = math_np.rotmat_abs_to_rel(abs_rotmat, sk.MANOHand.parents)
    assert(np.allclose(rel_rotmat, rel_rotmat_))
  print('Good.')


def layout_videos(src_paths, tar_path, height=512):
  readers = [vc.VideoReader(p) for p in src_paths]
  writer = None
  for _ in vc.progress_bar(range(readers[0].n_frames)):
    frames = [r.next_frame() for r in readers]
    done = np.any([f is None for f in frames])
    if done:
      break
    canvas = np.concatenate([vc.imresize_diag(f, h=height) for f in frames], 1)
    if writer is None:
      writer = vc.VideoWriter(
        tar_path, canvas.shape[1], canvas.shape[0], readers[0].fps
      )
    writer.write_frame(canvas)
  writer.close()
  [r.close() for r in readers]


if __name__ == '__main__':
  # test_math()
  # test_axangle_to_rotmat()
  # examine_selected_samples()
  # examine_mano_dataloader()
  # examine_mano_all_vert_dataloader()
  for video in os.listdir('output/yajiao_xyz'):
    if video.endswith('pkl'):
      video = video.replace('.pkl', '')
      layout_videos(
        [
          f'output/render/tips/{video}_mesh.mp4',
          f'output/render/{video}_uv.mp4',
          f'output/render/{video}_skeleton.mp4',
        ],
        f'output/render/tips/{video}_side.mp4'
      )
