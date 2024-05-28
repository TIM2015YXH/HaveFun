import torch
from diffusers import StableDiffusionPipeline
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


lpips = LPIPS(net_type='vgg')

pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base', torch_dtype=torch.float16, local_files_only=True)

print('down')