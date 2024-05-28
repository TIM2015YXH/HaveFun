import vctoolkit as vc
import torch
import os


class BaseConfig:
  def __init__(self):
    pass

  def as_dict(self):
    ret = {}
    for k, v in self.__dict__.items():
      if isinstance(v, BaseConfig):
        ret[k] = v.as_dict()
      else:
        ret[k] = v
    return ret


class GlobalConfig(BaseConfig):
  def __init__(self, path_file='paths_template.yaml'):
    self.mano_verts = 778
    self.mano_scale = 10.0
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.paths = {}
    for k, v in vc.load(path_file).items():
      self.paths[k] = v


GLOBAL = GlobalConfig()


class TrainConfig(BaseConfig):
  def __init__(self, name):
    super().__init__()
    self.name = name

    self.temporal = False

    self.log_every = 10
    self.log_dir = os.path.join(GLOBAL.paths['log_root'], name)

    self.eval_every = 1000
    self.latest_ckpt_path = \
      os.path.join(GLOBAL.paths['checkpoint_root'], name, 'latest.pth')
    self.best_ckpt_path = \
      os.path.join(GLOBAL.paths['checkpoint_root'], name, 'best.pth')
    self.tmp_ckpt_path = \
      os.path.join(GLOBAL.paths['checkpoint_root'], name, 'tmp.pth')

    self.debug = False

    self.init_lrate = 1e-3
    self.min_lrate = 1e-6
    self.loss_terms = {
      'pose': 1e1,
      'shape': 1e-3,
      'scale': 1e-2,
      'kpts': 1e1,
    }

    self.global_config = GLOBAL
