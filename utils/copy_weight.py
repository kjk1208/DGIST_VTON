import torch

# 체크포인트 파일 로드
paintnet_checkpoint_path = '../ckpts/paintnet_365_20240517.ckpt'
initial_checkpoint_path = '../ckpts/initial_checkpoint.ckpt'

# 체크포인트에서 state_dict 로드
paintnet_checkpoint = torch.load(paintnet_checkpoint_path, map_location='cpu')['state_dict']
initial_checkpoint = torch.load(initial_checkpoint_path, map_location='cpu')

# paintnet_365_20240517.ckpt에서 model.diffusion_model로 시작하는 모든 가중치를
# initial_checkpoint.ckpt 모델의 control_model로 복사
for key in list(paintnet_checkpoint.keys()):
    if key.startswith("model.diffusion_model"):
        # 새로운 키 이름을 생성 (예: model.diffusion_model.input -> control_model.input)
        new_key = "control_model" + key[len("model.diffusion_model"):]
        # initial_checkpoint의 새 키 위치에 가중치 복사
        initial_checkpoint[new_key] = paintnet_checkpoint[key]

# 수정된 체크포인트를 새 파일로 저장
torch.save(initial_checkpoint, initial_checkpoint_path.replace('initial_checkpoint.ckpt', 'updated_initial_checkpoint.ckpt'))

print("Weights have been copied and the updated checkpoint is saved.")