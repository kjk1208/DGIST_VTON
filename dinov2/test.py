import sys
import torch
import torch.nn as nn
sys.path.append("./dinov2")
import hubconf
from omegaconf import OmegaConf


config_path = '/home/kjk/DGIST_VTON/configs/DGIST.yaml'
config = OmegaConf.load(config_path)
#DINOv2_weight_path = config.model.params.cond_stage_config.weight
DINOv2_weight_path = '/home/kjk/DGIST_VTON/weight/dinov2_vitg14_pretrain.pth'

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
        self.projector = nn.Linear(1536,1024).to(device)

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
    

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to fit the input size of the model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def test_encoder(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenDinoV2Encoder(device=device)
    image_tensor = load_image(image_path)
    print("Image shape:", image_tensor.shape)
    
    # Encode image and get the output
    output = encoder.encode(image_tensor)
    
    # Print the output features
    print("Output features shape:", output.shape)
    print("Output features:", output)
    
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('runs/model_visualization')
    # writer.add_graph(encoder, torch.rand(1, 3, 224, 224))  # 임의의 입력으로 모델 구조 기록
    # writer.close()
    
    print(encoder)

    # Optional: Visualizing the output
    # For visualization, we need to process the output tensor. Here we just print it.
    # To visualize, you could use techniques such as reducing the dimensionality (e.g., using PCA) and plotting.

# Specify the path to your image
image_path = '/home/kjk/DGIST_VTON/DATA/VITON-HD/train/cloth/00000_00.jpg'
test_encoder(image_path)