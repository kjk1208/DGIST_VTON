import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def load_and_save_image(image_path, save_path):
    # 이미지 로딩 및 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # 이미지 텐서 정규화 해제
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_unnorm = unnormalize(image_tensor.squeeze(0)).clamp(0, 1)

    # 이미지 저장
    save_image(image_unnorm, save_path)
    print(f"Image saved to {save_path}")

# 이미지 경로
image_path = '/home/kjk/DGIST_VTON/DATA/VITON-HD/train/cloth/00000_00.jpg'
save_path = './image.jpg'

# 함수 호출
load_and_save_image(image_path, save_path)