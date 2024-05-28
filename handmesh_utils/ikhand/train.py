import network
import data
import torch
import os
import vctoolkit as vc
import skeletons as sk
import config_xchen as cfg
from ikhand import utils as utils
from torch.utils.tensorboard import SummaryWriter
import math_pt
import numpy as np
import evaluation
from torch.utils.data import DataLoader


def train(model_name, model, datasets, settings, val_dataloader):
  eps = torch.tensor(np.finfo(np.float32).eps).to(cfg.DEVICE)
  model_dir = os.path.join(cfg.CKPT_DIR, model_name)
  os.makedirs(model_dir, exist_ok=True)

  timer = vc.Timer()
  summary_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, model_name))
  optimizer = torch.optim.Adam(model.parameters(), lr=settings['lrate'])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=settings['patience'])

  step = utils.restore_state(model_name, model, optimizer, scheduler, settings)
  vc.save(os.path.join(model_dir, f'settings_{step}.json'), settings)

  iterations = {k: iter(v['dataloader']) for k, v in datasets.items()}

  while optimizer.param_groups[0]["lr"] > settings['min_lrate']:
    batch_input = utils.collect_batch_input(iterations, datasets)
    batch_output = model(batch_input['inputs'].to(cfg.DEVICE))
    batch_output['keypoints'] = math_pt.rot6d_fk(
      batch_output['pose'].view(-1, sk.MANOHand.n_joints, 6),
      batch_input['ref_bones'].to(cfg.DEVICE), sk.MANOHand, eps
    )

    loss_terms = {}
    for k, v in settings['loss_terms'].items():
      gt = batch_input[k].to(cfg.DEVICE)
      pred = batch_output[k].to(cfg.DEVICE)
      mse = torch.mean(torch.square(gt - pred))
      loss_terms[k] = mse * v
    loss = torch.sum(torch.stack(list(loss_terms.values())))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1

    print(
      f'{model_name}: step {step} ' \
      f'loss {loss.item():5e} ' \
      f'time {int(timer.tic() * 1000):d}ms ' \
      f'lrate {optimizer.param_groups[0]["lr"]}'
    )

    if step % cfg.LOG_EVERY == 0:
      summary_writer.add_scalar('loss/overall_loss', loss, step)
      for k, v in loss_terms.items():
        summary_writer.add_scalar(f'loss_terms/{k}', v, step)
      error = torch.mean(
        torch.linalg.norm(
          batch_input['keypoints'].to(cfg.DEVICE) - batch_output['keypoints'], axis=-1
        )
      )
      summary_writer.add_scalar('error/keypoints', error, step)

    if step % cfg.SAVE_EVERY == 0:
      model_path = os.path.join(model_dir, f'{step}.pth')
      torch.save(
        {
          'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(), 'step': step
        },
        model_path
      )
      vc.save_json(
        os.path.join(model_dir, cfg.LATEST_MODEL_INFO_FILE),
        {'path': model_path}
      )
      validation_mse = evaluation.eval_on_mano_data(model, val_dataloader)
      scheduler.step(validation_mse)
      summary_writer.add_scalar(f'validation/keypoint', validation_mse, step)


def train_example():
  if cfg.DEVICE == 'cpu':
    for _ in range(10):
      print('===== WARNING: device is cpu. =====')

  model_name = 'iknet_conv'

  settings = {
    'lrate': 1e-3,
    'min_lrate': 1e-6,
    'patience': 10,

    'depth': 4,
    'width': 1024,

    'mano_dataloader': {
      'batch_size': 1024,
      'shape_std': 1,
      'scale_std': 0.1,
      'conf_mean': 0.9,
      'conf_std': 0.1,
      'noise_std': 0.1,
      'p_power': 0,
      'mix_fingers': False,
    },

    'loss_terms': {
      'pose': 1e1,
      'shape': 1e-3,
      'scale': 1e-2,
      'keypoints': 1e1
     },
  }

  datasets = {
    'mano': {
      'dataloader': data.InfiniteDataLoader(
        data.MANODatasetKwaiWrapper(
          data.MANODatasetINTRP(settings['mano_dataloader'])
        ),
        batch_size=1, shuffle=False, num_workers=1
      )
    }
  }

  val_dataloader = DataLoader(
    data.MANODatasetKwaiWrapper(data.MANODatasetOriginal())
  )

  model = network.IKNetConvV1(
    sk.KWAIHand.n_keypoints * 3, sk.MANOHand.n_joints * 6,
    shape_dim=10, depth=settings['depth'], width=settings['width']
  ).to(cfg.DEVICE)

  train(model_name, model, datasets, settings, val_dataloader)


if __name__ == '__main__':
  train_example()
