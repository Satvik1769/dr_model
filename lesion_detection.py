from PIL import Image
from torchvision import transforms
from gradcam import visualize_gradcam
from model_architecture.dense_net import model
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('./models/densenet_dr_model.pth', map_location=device))
model.eval().to(device)

img_path = "./gaussian_filtered_images/Moderate/0180bfa26c0b.png"
img = Image.open(img_path).convert("RGB")
model_name = type(model).__name__.lower()

if 'vit' not in model_name:
    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
])
else:

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
    ])

image_tensor = transform(img)


# Define the DR class labels
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
def get_target_layer(model):
    model_name = type(model).__name__.lower()
    print(model_name)

    if 'densenet' in model_name:
        return 'features.denseblock4'
    elif 'resnetcbam' in model_name:
        for name, _ in model.named_modules():
            print(name)
        return 'model.layer4'
    elif 'resnet' in model_name:
        return 'layer4'
    elif 'efficientnet' in model_name:
        return 'features'
    elif 'inception' in model_name:
        return 'Mixed_7c'
    elif 'vit_cbam' in model_name:
        return 'cbam.spatial_attention.conv'
    elif 'vit' in model_name:
        return 'blocks.11'
    elif 'cnn' in model_name:
        return 'features.6'
    elif 'attention' in model_name:
        # Try to hook into the last conv inside self.base_model
        for name, module in model.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = name
        if last_conv:
            return f'base_model.{last_conv}'  # full path to hook
        else:
            raise RuntimeError("No Conv2d layer found inside base_model")
    else:
        raise ValueError(f"Unknown model type: {model_name}")


target_layer = get_target_layer(model=model)

# Show Grad-CAM
visualize_gradcam(image_tensor, class_names, model, target_layer=target_layer)
