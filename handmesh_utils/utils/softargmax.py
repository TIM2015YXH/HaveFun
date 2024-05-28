import torch
import torch.nn as nn


def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim() == 5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 10000.0
    N, C, H, W, D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N, C, -1)*alpha, dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0, end=H * W * D).unsqueeze(0)
    indices_kernel = indices_kernel.view((H, W, D))
    conv = soft_max * indices_kernel.to(soft_max.device)
    indices = conv.sum(2).sum(2).sum(2)
    z = indices % D
    y = (indices / D) % W
    x = (((indices / D)) / W) % H
    coords = torch.stack([x, y, z], dim=2)
    return coords


if __name__ == "__main__":
    voxel = torch.randn(1, 17, 128, 128, 1) # (batch_size, channel, H, W, depth)
    coords = soft_argmax(voxel)
    print(coords, coords.shape)
