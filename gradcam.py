import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_gradcam(model, image_tensor,target_layer, class_idx=None):
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks to the target layer
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break

    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad = True
    output = model(image_tensor)

    if isinstance(output, tuple):
        output = output[0]  # main output

    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    score = output[0, class_idx]
    score.backward()

    gradient = gradients[0]
    activation = activations[0]

    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activation, dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cam.cpu().numpy()
    model_name = type(model).__name__.lower()
    if 'vit_cbam' not in model_name:
        cam = cv2.resize(cam, (299, 299))
    else:
        cam = cv2.resize(cam, (224, 224))

    return cam, class_idx






def visualize_gradcam(image_tensor, class_names, model, target_layer):
    cam, class_idx = generate_gradcam(model, image_tensor, target_layer)
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    overlay = heatmap * 0.4 + img

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM - Class: {class_names[class_idx]}")
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()