from vctoolkit import viso3d as v3d
import config as cfg
import math_np
import numpy as np
import vctoolkit as vc
import mesh
import utils
import os
import skeletons as sk


def render_hand_mesh(src_data, tar_path, view_mat, vert_smoother=None):
  mano = mesh.MeshModel(cfg.MANO_PATH)
  shape = np.mean([v['shape'] for v in src_data.values()], axis=0)
  scale = np.mean([v['scale'] for v in src_data.values()], axis=0)
  v_seq = []
  for _, v in vc.progress_bar(src_data.items()):
    rot6d = np.reshape(v['pose'], [21, 6])
    rotmat = math_np.convert(rot6d, 'rot6d', 'rotmat')
    _, v = mano.set_params(rotmat, beta=shape)
    v *= scale
    v, f = utils.left_right_flip(v, mano.faces)
    v = np.einsum('hw, nw -> nh', view_mat, v)
    if vert_smoother is not None:
      v = vert_smoother.process(v)
    v_seq.append(v)
  v3d.render_sequence_3d(v_seq, f, 1024, 1024, tar_path)


def render_hand_xyz(src_data, tar_path):
  v_seq = []
  for _, v in vc.progress_bar(src_data.items()):
    xyz = utils.centralize_batch([v['xyz']], sk.KWAIHand.center)[0]
    v, f = vc.joints_to_mesh(xyz, sk.KWAIHand.parents)
    v, f = utils.left_right_flip(v, f)
    v = np.einsum('hw, nw -> nh', VIEW_MAT, v)
    v_seq.append(v)
  v3d.render_sequence_3d(v_seq, f, 1024, 1024, tar_path)


def render_hand_uv(src_video, src_data, tar_path):
  reader = vc.VideoReader(src_video)
  writer = vc.VideoWriter(tar_path, reader.width, reader.height, reader.fps)
  for _, v in vc.progress_bar(src_data.items()):
    canvas = reader.next_frame()
    if canvas is None:
      break
    uv = v['uv'].T.copy()
    uv[:, 1] = reader.width - uv[:, 1]
    vc.render_bones_from_uv(uv, canvas, sk.KWAIHand.parents, sk.KWAIHand.colors)
    writer.write_frame(canvas)
  reader.close()
  writer.close()


if __name__ == '__main__':
  src_dir = 'output/interhand'
  tar_dir = 'output/render/interhand_tips'
  view_mat = np.dot(
    math_np.convert(np.array([0, 0, np.pi/2]), 'axangle', 'rotmat'),
    math_np.convert(np.array([np.pi/2, 0, 0]), 'axangle', 'rotmat'),
  )
  os.makedirs(tar_dir, exist_ok=True)
  for video in os.listdir(src_dir):
    if video.endswith('pkl'):
      video = video.replace('.pkl', '')
      render_hand_mesh(
        vc.load(os.path.join(src_dir, video + '.pkl')),
        os.path.join(tar_dir, video + '_mesh.mp4'),
        view_mat
      )
      # render_hand_xyz(
      #   vc.load(f'input/yajiao_xyz/{video}.pkl'),
      #   f'output/render/{video}_skeleton.mp4',
      # )
      # render_hand_uv(
      #   f'input/xingyu_videos/{video}.mov',
      #   vc.load(f'input/yajiao_xyz/{video}.pkl'),
      #   f'output/render/{video}_uv.mp4',
      # )
