import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from omegaconf import OmegaConf
sys.path.append("./dinov2")
import hubconf

config_path = './configs/DGIST.yaml'
config = OmegaConf.load(config_path)
DINOv2_weight_path = config.model.params.cond_stage_config.weight

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitg14() 
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        #self.projector = nn.Linear(1536,1024)
        self.projector = nn.Linear(1536,768)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features,tokens],1) # 8,257,1024
        hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)
    
if __name__ == "__main__":
    import os
    from PIL import Image

    image_path = "/home/kjk/DGIST_VTON/DATA/VITON-HD/train/cloth/00009_00.jpg"
    output_path = "./dinov2_pca_output.png"

    image = Image.open(image_path).convert("RGB")

    # 32x24 입력 비율 맞춤
    transform = transforms.Compose([
        transforms.Resize((384, 512)),  # 16x32 = 512, 16x24 = 384
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 384, 512]

    encoder = FrozenDinoV2Encoder(device="cpu")
    encoder.eval()

    with torch.no_grad():
        hint = encoder.encode(image_tensor.to("cpu"))  # [1, 769, 1024]

    patch_features = hint[0, 1:, :]  # [768, 1024]

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(patch_features.cpu().numpy())  # [768, 3]
    reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min())

    # 32 x 24 patch 시각화
    color_map = reduced.reshape(24, 32, 3)  # H=24, W=32
    color_map = np.uint8(color_map * 255)

    # 시각화 후 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.resize((512, 384)))  # 입력 이미지 크기 맞춰서 시각화
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(color_map)
    plt.title("DINOv2 Patch PCA (RGB)")

    plt.savefig(output_path)
    print(f"✅ 시각화 결과 저장됨: {os.path.abspath(output_path)}")