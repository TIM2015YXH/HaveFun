import config as cfg
import network
import skeletons as sk
import os
import vctoolkit as vc
import torch
import utils
import numpy as np
import data
import torch
import math_np
import math_pt
import mesh
import math_np


def eval_on_kwai_data(xyz_smoother=None):
  mano_size = math_np.measure_hand_size(
    utils.load_official_mano_model(cfg.MANO_PATH)['keypoints_mean'],
    sk.MANOHand
  ) * cfg.MANO_SCALE

  model = network.IKNetV1(
    sk.KWAIHand.n_keypoints * 3, sk.MANOHand.n_joints * 6,
    shape_dim=10, depth=4, width=1024
  ).to(cfg.DEVICE)
  model.load_state_dict(
    torch.load(
      os.path.join('model', 'iknet_v9.3', '48600.pth'),
      map_location=torch.device(cfg.DEVICE)
    )['model']
  )
  model.eval()

  output_dir = 'output/yajiao_xyz'
  os.makedirs(output_dir, exist_ok=True)

  with torch.no_grad():
    for video in vc.progress_bar(os.listdir(cfg.TEST_DATA_1_DIR), 'videos'):
      detection = vc.load(os.path.join(cfg.TEST_DATA_1_DIR, video))
      ik = {}
      for frame in vc.progress_bar(detection, 'frames'):
        xyz = detection[frame]['xyz'].astype(np.float32)
        xyz = utils.centralize_batch([xyz], sk.KWAIHand.center)[0]

        scale = mano_size / math_np.measure_hand_size(xyz, sk.KWAIHand)
        xyz *= scale
        if xyz_smoother is not None:
          xyz = xyz_smoother.process(xyz)

        pred = model(torch.from_numpy(np.reshape(xyz, [1, -1])).to(cfg.DEVICE))
        ik[frame] = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
      vc.save(os.path.join(output_dir, video), ik)


def eval_on_kwai_data_tips(xyz_smoother=None):
  ref_keypoints = \
    utils.load_official_mano_model(cfg.MANO_PATH)['keypoints_mean']
  mano_size = \
    math_np.measure_hand_size(ref_keypoints, sk.MANOHand) * cfg.MANO_SCALE
  ref_bones = \
    math_np.keypoints_to_bones_batch([ref_keypoints], sk.MANOHand.parents)[0]
  mano_bone_lengths = np.linalg.norm(ref_bones, axis=-1) * cfg.MANO_SCALE

  model = network.IKNetV1(
    39, sk.MANOHand.n_joints * 6,
    shape_dim=10, depth=4, width=2048
  ).to(cfg.DEVICE)
  model.load_state_dict(
    torch.load(
      os.path.join('model', 'iknet_tips', '14300.pth'),
      map_location=torch.device(cfg.DEVICE)
    )['model']
  )
  model.eval()

  output_dir = 'output/yajiao_xyz_tips'
  os.makedirs(output_dir, exist_ok=True)

  with torch.no_grad():
    for video in vc.progress_bar(os.listdir(cfg.TEST_DATA_1_DIR), 'videos'):
      detection = vc.load(os.path.join(cfg.TEST_DATA_1_DIR, video))
      ik = {}
      for frame in vc.progress_bar(detection, 'frames'):
        xyz = detection[frame]['xyz'].astype(np.float32)
        xyz = utils.centralize(xyz, sk.KWAIHand.center)

        scale = mano_size / math_np.measure_hand_size(xyz, sk.KWAIHand)
        xyz *= scale
        if xyz_smoother is not None:
          xyz = xyz_smoother.process(xyz)
        xyz = xyz[sk.KWAIHand.end_points]
        bone_lengths = mano_bone_lengths * scale
        inputs = np.concatenate([xyz.ravel(), bone_lengths], -1)
        pred = model(torch.from_numpy(np.reshape(inputs, [1, -1])).to(cfg.DEVICE))
        ik[frame] = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
      vc.save(os.path.join(output_dir, video), ik)


def eval_on_fake_data():
  N = 10240
  model = network.IKNetV1(
    sk.KWAIHand.n_keypoints * 3, sk.MANOHand.n_joints * 6,
    shape_dim=10, depth=6, width=1024
  ).to(cfg.DEVICE)
  model.load_state_dict(
    torch.load(os.path.join(cfg.CKPT_DIR, 'iknet_v6', '1500.pth')
  )['model'])
  model.eval()

  mano = mesh.MeshModel(cfg.MANO_PATH)

  np.random.seed(1019)

  setting = {
    'shape_std': 1,
    'scale_std': 0.1,
    'conf_mean': 1.0,
    'conf_std': 0.0,
    'noise_std': 0.0,
    'p_power': 2,
    'mix_fingers': False,
  }

  dataset = data.MANODatasetKwaiWrapper(data.MANODatasetINTRPL(setting))

  with torch.no_grad():
    for i in range(N):
      pack = dataset[i]
      inputs = torch.from_numpy(np.expand_dims(pack['inputs'], 0)).to(cfg.DEVICE)
      pred = {k: v.detach().cpu().numpy()[0] for k, v in model(inputs).items()}
      gt = pack['keypoints']

      abs_rot6d = np.reshape(pred['pose'], [-1, 6])
      shape = pred['shape']
      scale = pred['scale']
      abs_rotmat = math_np.convert(abs_rot6d, 'rot6d', 'rotmat')
      k, v = mano.set_params(abs_rotmat, shape)
      vc.save_obj(f'test_mesh_pred_{i}.obj', v * 10, mano.faces)
      k = utils.centralize(k, sk.MANOHand.labels.index('M0')) * cfg.MANO_SCALE * scale
      gt = utils.centralize(gt, sk.MANOHand.labels.index('M0'))

      v, f = vc.joints_to_mesh(k, sk.MANOHand.parents)
      vc.save_obj(f'test_ik_pred_{i}.obj', v, f)
      v, f = vc.joints_to_mesh(gt, sk.MANOHand.parents)
      vc.save_obj(f'test_ik_gt_{i}.obj', v, f)

      mse = np.linalg.norm(k - gt)
      print(mse)

      vc.press_to_continue()


def eval_on_fake_data_tips():
  model = network.IKNetV1(
    39, sk.MANOHand.n_joints * 6, shape_dim=10, depth=4, width=2048
  ).to(cfg.DEVICE)
  model.load_state_dict(
    torch.load(os.path.join('model', 'iknet_tips', '14300.pth'),
    map_location=torch.device(cfg.DEVICE)
  )['model'])
  model.eval()

  mano = mesh.MeshModel(cfg.MANO_PATH)

  np.random.seed(1019)

  setting = {
    'batch_size': 8,
    'shape_std': 1,
    'scale_std': 0.1,
    'conf_mean': 1.0,
    'conf_std': 0.0,
    'noise_std': 0.0,
    'p_power': 0,
    'mix_fingers': False,
  }

  dataset = data.MANODatasetTipsWrapper(data.MANODatasetINTRP(setting))
  pack_batch = dataset[0]

  with torch.no_grad():
    for i in range(setting['batch_size']):
      pack = {k: v[i] for k, v in pack_batch.items()}
      inputs = torch.from_numpy(np.expand_dims(pack['inputs'], 0)).to(cfg.DEVICE)
      pred = {k: v.detach().cpu().numpy()[0] for k, v in model(inputs).items()}
      gt = pack['keypoints']

      abs_rot6d = np.reshape(pred['pose'], [-1, 6])
      shape = pred['shape']
      scale = pred['scale']
      abs_rotmat = math_np.convert(abs_rot6d, 'rot6d', 'rotmat')
      k, v = mano.set_params(abs_rotmat, shape)
      vc.save_obj(f'test_mesh_pred_{i}.obj', v * 10, mano.faces)
      k = utils.centralize(k, sk.MANOHand.labels.index('M0')) * cfg.MANO_SCALE * scale
      gt = utils.centralize(gt, sk.MANOHand.labels.index('M0'))

      v, f = vc.joints_to_mesh(k, sk.MANOHand.parents)
      vc.save_obj(f'test_ik_pred_{i}.obj', v, f)
      v, f = vc.joints_to_mesh(gt, sk.MANOHand.parents)
      vc.save_obj(f'test_ik_gt_{i}.obj', v, f)

      mse = np.linalg.norm(k - gt)
      print(mse)

      vc.press_to_continue()


def eval_on_interhand_tips():
  model = network.IKNetV1(
    39, sk.MANOHand.n_keypoints * 6, shape_dim=10, depth=4, width=2048
  ).to(cfg.DEVICE)
  model.load_state_dict(
    torch.load(
      os.path.join('model/iknet_tips/14300.pth'),
      map_location=torch.device(cfg.DEVICE)
    )['model']
  )
  model.eval()

  data_path = '/Users/yzhou/Documents/Workspace/hand-mesh-pointcloud-tracking/pchand/data/interhand/dump/train_0_000004_381.pkl'
  data = vc.load(data_path)
  kpt_seq = np.array([
    utils.centralize(d['left']['keypoints'], sk.MANOHand.center) for d in data
  ])
  tip_seq = kpt_seq[:, sk.MANOHand.end_points]
  bone_seq = math_np.keypoints_to_bones_batch(kpt_seq, sk.MANOHand.parents)
  len_seq = np.linalg.norm(bone_seq, axis=-1)
  N = tip_seq.shape[0]
  inputs = np.concatenate(
    [np.reshape(tip_seq, [N, -1]), np.reshape(len_seq, [N, -1])], -1
  )
  with torch.no_grad():
    pred = model(torch.from_numpy(inputs.astype(np.float32)))
  pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
  output = {}
  for i in range(pred['pose'].shape[0]):
    output[str(i)] = {k: v[i] for k, v in pred.items()}
  tar_dir = 'output/interhand'
  os.makedirs(tar_dir, exist_ok=True)
  vc.save(os.path.join(tar_dir, os.path.basename(data_path)), output)


def eval_on_mano_data(model, dataloader):
  eps = torch.tensor(np.finfo(np.float32).eps).to(cfg.DEVICE)
  model.eval()
  with torch.no_grad():
    residual = []
    for batch_input in dataloader:
      batch_input = {k: v[0] for k, v in batch_input.items()}
      batch_output = model(batch_input['inputs'].to(cfg.DEVICE))
      keypoints_pred = math_pt.rot6d_fk(
        batch_output['pose'].view(-1, sk.MANOHand.n_joints, 6),
        batch_input['ref_bones'].to(cfg.DEVICE), sk.MANOHand, eps
      ).to(cfg.DEVICE)
      keypoints_gt = batch_input['keypoints'].to(cfg.DEVICE)
      residual.append(keypoints_gt - keypoints_pred)
    residual = torch.cat(residual)
    error = torch.mean(torch.linalg.norm(residual, axis=-1))
  model.train()
  return error


if __name__ == '__main__':
  # eval_on_kwai_data(xyz_smoother=vc.OneEuroFilter(2.0))
  # eval_on_fake_data_tips()
  # eval_on_kwai_data_tips()
  eval_on_interhand_tips()
