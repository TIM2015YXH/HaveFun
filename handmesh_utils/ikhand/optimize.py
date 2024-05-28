import math_np
import numpy as np
import vctoolkit as vc
import cv2
import mesh
import config as cfg
import utils
import skeletons as sk


class PnPSolver:
  def __init__(self, f, c, reset_threshold=1.5):
    if not isinstance(c, list):
      c = [c, c]
    self.cam_mat = np.array([
      [f, 0, c[0]],
      [0, -f, c[1]],
      [0, 0, 1]
    ], np.float32)
    self.reset_threshold = reset_threshold
    self.need_reset = True
    self.prev_axangle = np.zeros(3, dtype=np.float32)
    self.prev_trans = np.zeros(3, dtype=np.float32)

  def solve(self, xyz, uv):
    # uv should be [col, row]
    _, axangle, trans = cv2.solvePnP(
      xyz, uv, self.cam_mat, np.zeros([4, 1], np.float32),
      self.prev_axangle, self.prev_trans,
      useExtrinsicGuess=(not self.need_reset)
    )
    axangle = np.reshape(axangle, [3])
    trans = np.reshape(trans, [1, 3])
    rotmat = math_np.convert(axangle, 'axangle', 'rotmat')
    xyz_glb = np.einsum('hw, nw -> nh', rotmat, xyz) + trans
    uv_proj = vc.camera_proj(self.cam_mat, xyz_glb)
    mse = np.mean(np.linalg.norm(uv_proj - uv, axis=-1))
    rate = mse / np.max(np.max(uv, axis=0) - np.min(uv, axis=0))
    if rate > self.reset_threshold:
      self.need_reset = True
    else:
      self.need_reset = False
    return rotmat, trans, xyz_glb, uv_proj


if __name__ == '__main__':
  mano = mesh.MeshModel(cfg.MANO_PATH)
  video = 'IMG_1170'
  pnp_solver = PnPSolver(cfg.CAM_F, cfg.CAM_C)
  det_data = vc.load(f'input/yajiao_xyz/{video}.pkl')
  ik_data = vc.load(f'output/yajiao_xyz/{video}.pkl')
  shape = np.mean([v['shape'] for v in ik_data.values()], axis=0)
  scale = np.mean([v['scale'] for v in ik_data.values()], axis=0)
  reader = vc.VideoReader(f'input/xingyu_videos/{video}.mov')
  writer = None
  for k in vc.progress_bar(ik_data):
    canvas = reader.next_frame()
    rot6d = np.reshape(ik_data[k]['pose'], [21, 6])
    rotmat = math_np.convert(rot6d, 'rot6d', 'rotmat')
    xyz = mano.set_params(rotmat, beta=shape)[0].astype(np.float32)
    xyz[:, 0] *= -1 # from left hand to right hand
    xyz = utils.convert_skeleton_batch(
      [xyz], sk.MANOHand.labels, sk.KWAIHand.labels
    )[0]
    xyz = utils.centralize_batch([xyz], sk.KWAIHand.center)[0]
    xyz *= cfg.MANO_SCALE # from meter to decimeter
    uv = det_data[k]['uv'].T.copy().astype(np.float32)
    r, t, xyz_glb, uv_proj = pnp_solver.solve(xyz, uv)

    uv[:, 1] = canvas.shape[1] - uv[:, 1]
    uv_proj[:, 1] = canvas.shape[1] - uv_proj[:, 1]
    a = vc.render_bones_from_uv(
      uv, canvas.copy(), sk.KWAIHand.parents, sk.KWAIHand.colors
    )
    b = vc.render_bones_from_uv(
      uv_proj, canvas.copy(), sk.KWAIHand.parents, sk.KWAIHand.colors
    )
    frame = np.concatenate([a, b], 1)
    if writer is None:
      writer = vc.VideoWriter(
        f'./{video}_pnp.mp4', frame.shape[1], frame.shape[0], reader.fps
      )
    writer.write_frame(frame)
  writer.close()
  reader.close()
