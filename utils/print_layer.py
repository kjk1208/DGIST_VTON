import torch

# Load the checkpoint
checkpoint_path = '../ckpts/VITONHD_modified.ckpt'
checkpoint = torch.load(checkpoint_path)

# Access the state dictionary
state_dict = checkpoint

# Print all layer keys
for key in state_dict.keys():
    print(key)
    