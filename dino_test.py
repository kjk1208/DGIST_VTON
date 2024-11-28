import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# DINO v2의 GitHub 리포지토리에서 필요한 모듈을 로드합니다.
import sys
sys.path.append('/home/kjk/DGIST_VTON/dinov2')

import hubconf

DINOv2_weight_path = '/home/kjk/DGIST_VTON/weight/dinov2_vitg14_pretrain.pth'
#from dinov2 import DinoImageEncoder

# 이미지를 처리할 디바이스 설정 (GPU 사용 가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 인코더 초기화
#encoder = DinoImageEncoder()
dinov2 = hubconf.dinov2_vitg14() 
dinov2.to(device)

# 이미지 전처리를 위한 트랜스폼 설정
transform = T.Compose([
    T.Resize((224, 224)),  # DINO v2의 입력 사이즈에 맞추기
    T.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def encode_and_visualize(image_path):
    # 이미지 로드 및 전처리
    image_tensor = load_and_preprocess_image(image_path).to(device)

    # 이미지 인코딩
    with torch.no_grad():
        encoded_image = dinov2(image_tensor)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.cpu().squeeze().permute(1, 2, 0).numpy())
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # 인코딩 결과 시각화 (여기서는 간단히 시각화 예를 들었습니다)
    # 인코딩된 이미지는 특성 맵이므로, 이를 시각화하기 위해 간단한 처리가 필요할 수 있습니다.
    plt.imshow(encoded_image.cpu().squeeze()[0], cmap='viridis')
    plt.title("Encoded Image")
    plt.axis('off')

    plt.show()

# 이미지 경로를 입력받습니다.
image_path = input("/home/kjk/DGIST_VTON/DATA/VITON-HD/train/cloth/00000_00.jpg")
encode_and_visualize(image_path)
