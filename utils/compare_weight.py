import torch

# 두 개의 체크포인트 파일 로드
checkpoint_path1 = '../ckpts/initial_checkpoint.ckpt'
checkpoint_path2 = '../ckpts/VITONHD.ckpt'
checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')
checkpoint2 = torch.load(checkpoint_path2, map_location='cpu')

# 각 체크포인트의 state_dict 접근
state_dict1 = checkpoint1
state_dict2 = checkpoint2

# 같은 이름의 레이어에서 가중치 값이 같은지 확인
same_weights = []
different_weights = []
different_layer = []

for key in state_dict1.keys():
    if key in state_dict2:
        # 레이어가 두 state_dict 모두에 존재하면 가중치 비교
        if torch.equal(state_dict1[key], state_dict2[key]):
            same_weights.append(key)
        else:
            different_weights.append(key)
    else:
        # 두 번째 state_dict에 해당 키가 없으면, 다른 가중치 리스트에 추가
        different_layer.append(key)

# 결과 출력
print("Layers with same weights:")
for layer in same_weights:
    print(layer)

print("\nLayers with different weights:")
for layer in different_weights:
    print(layer)

print("\nLayers with different layers:")
for layer in different_layer:
    print(layer)