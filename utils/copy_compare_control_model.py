import torch

# 체크포인트 파일 로드
updated_checkpoint_path = '../ckpts/updated_initial_checkpoint.ckpt'
paintnet_checkpoint_path = '../ckpts/paintnet_365_20240517.ckpt'

# 체크포인트에서 state_dict 로드
updated_checkpoint = torch.load(updated_checkpoint_path, map_location='cpu')
paintnet_checkpoint = torch.load(paintnet_checkpoint_path, map_location='cpu')['state_dict']

# 각 체크포인트의 state_dict 접근
state_dict_updated = updated_checkpoint
state_dict_paintnet = paintnet_checkpoint

# 가중치 비교를 위한 리스트 초기화
same_weights = []
different_weights = []

# updated_initial_checkpoint의 control_model 가중치와 paintnet_365_20240517의 model.diffusion_model 가중치 비교
for key in state_dict_updated.keys():
    if key.startswith("control_model"):
        # paintnet 체크포인트에서 대응하는 키 이름 변환
        corresponding_key = "model.diffusion_model" + key[len("control_model"):]

        if corresponding_key in state_dict_paintnet:
            # 두 state_dict에서 가중치 비교
            if torch.equal(state_dict_updated[key], state_dict_paintnet[corresponding_key]):
                same_weights.append(key)
            else:
                different_weights.append(key)

# 결과를 out.txt 파일로 출력
output_path = './out.txt'
with open(output_path, 'w') as f:
    f.write("Layers with same weights:\n")
    for layer in same_weights:
        f.write(layer + '\n')

    f.write("\nLayers with different weights:\n")
    for layer in different_weights:
        f.write(layer + '\n')

print(f"Results have been saved to {output_path}")