import torch.nn as nn


class DenseBN(nn.Module):
  def __init__(self, in_width, out_width, activation=nn.ReLU):
    super().__init__()
    self.block = nn.Sequential(
      nn.Linear(in_width, out_width),
      nn.BatchNorm1d(out_width)
    )
    if activation is not None:
      self.activation = activation()
    else:
      self.activation = None

  def forward(self, x):
    x = self.block(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


class DenseBN_Conv(nn.Module):
  def __init__(self, in_width, out_width, activation=nn.ReLU):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_width, out_width, 1, 1, 0, bias=False),
      nn.BatchNorm2d(out_width)
    )
    if activation is not None:
        self.activation = activation()
    else:
        self.activation = None

  def forward(self, x):
    x = self.block(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


class IKNetV1(nn.Module):
  def __init__(self, input_dim, pose_dim, shape_dim, depth, width):
    super().__init__()
    self.encoder = DenseBN(input_dim, width)
    self.layers = nn.Sequential(*[DenseBN(width, width) for _ in range(depth)])
    self.pose_decoder = nn.Linear(width, pose_dim)
    self.shape_decoder = nn.Linear(width, shape_dim)
    self.scale_decoder = nn.Linear(width, 1)

  def forward(self, x):
    x = self.encoder(x)
    x = self.layers(x)
    pose = self.pose_decoder(x)
    shape = self.shape_decoder(x)
    scale = self.scale_decoder(x)
    pack = {'pose': pose, 'shape': shape, 'scale': scale}
    return pack


class IKNetConvV1(nn.Module):
  def __init__(self, input_dim, pose_dim, shape_dim, depth, width):
    super().__init__()
    self.encoder = DenseBN_Conv(input_dim, width)
    self.layers = nn.Sequential(*[DenseBN_Conv(width, width) for _ in range(depth)])
    self.pose_decoder = nn.Conv2d(width, pose_dim, 1, 1, 0, bias=False)
    self.shape_decoder = nn.Conv2d(width, shape_dim, 1, 1, 0, bias=False)
    self.scale_decoder = nn.Conv2d(width, 1, 1, 1, 0, bias=False)
    self.pose_dim = pose_dim
    self.shape_dim = shape_dim

  def forward(self, x):
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = self.encoder(x)
    x = self.layers(x)
    pose = self.pose_decoder(x)
    shape = self.shape_decoder(x)
    scale = self.scale_decoder(x)
    pose = pose.view(-1, self.pose_dim)
    shape = shape.view(-1, self.shape_dim)
    scale = scale.view(-1, 1)
    pack = {'pose': pose, 'shape': shape, 'scale': scale}
    return pack
