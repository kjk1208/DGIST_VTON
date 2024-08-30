import torch
checkpoint = torch.load('../ckpts/VITONHD.ckpt', map_location='cpu')
print(checkpoint.keys())
