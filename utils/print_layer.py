# import torch

# # Load the checkpoint
# #checkpoint_path = '../ckpts/VITONHD_modified.ckpt'
# checkpoint_path = '../ckpts/paintnet_365_20240517.ckpt'
# checkpoint = torch.load(checkpoint_path)

# # Access the state dictionary
# state_dict = checkpoint['state_dict']

# # Print all layer keys
# for key in state_dict.keys():
#     print(key)
    

# with open('model.txt', 'w') as file:
#     # Iterate through the keys and write them to the file
#     for key in state_dict.keys():
#         file.write(key + '\n')

# print("Keys have been saved to model.txt")


import torch
import os

# Load the checkpoint
#checkpoint_path = '../ckpts/VITONHD_modified.ckpt'
#checkpoint_path = '../ckpts/paintnet_365_20240517.ckpt'
checkpoint_path = '../ckpts/initial_checkpoint.ckpt'
checkpoint = torch.load(checkpoint_path)

# Access the state dictionary
state_dict = checkpoint
#state_dict = checkpoint['state_dict']

# Generate the output file name based on the checkpoint path
output_file_name = os.path.splitext(os.path.basename(checkpoint_path))[0] + '.txt'
output_file_path = os.path.join(os.path.dirname(checkpoint_path), output_file_name)

# for key in state_dict.keys():
#     print(key)

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Iterate through the keys and write them to the file
    for key in state_dict.keys():
        file.write(key + '\n')

print(f"Keys have been saved to {output_file_path}")
