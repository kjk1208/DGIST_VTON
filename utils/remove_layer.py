import torch

# Checkpoint 파일 로드
checkpoint_path = '../ckpts/VITONHD_modified.ckpt'
checkpoint = torch.load(checkpoint_path)

# state_dict 접근
state_dict = checkpoint

# 특정 레이어 키 필터링 및 삭제
layer_prefix = "model.diffusion_model.warp_zero_convs"
keys_to_delete = [key for key in state_dict.keys() if key.startswith(layer_prefix)]

for key in keys_to_delete:
    del state_dict[key]

# 수정된 checkpoint 저장
new_checkpoint_path = '../ckpts/VITONHD_modified.ckpt'
torch.save(state_dict, new_checkpoint_path)

print(f"Deleted keys: {keys_to_delete}")
    
    