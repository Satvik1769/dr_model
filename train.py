import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from model_architecture.vit_cbam import model, train

# -------- Custom Dataset --------
custom_class_to_idx = {
    'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4
}

if type(model).__name__.lower() == 'vit_cbam':
    from model_architecture.vit_cbam import train_transform, val_transform 
else:
    train_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = custom_class_to_idx
        self.samples = [(path, self.class_to_idx[os.path.basename(os.path.dirname(path))])
                        for path, _ in self.samples]
        self.targets = [s[1] for s in self.samples]

# -------- Training Setup --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



dataset = CustomImageFolder(root='gaussian_filtered_images/', transform=train_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

labels = dataset.targets
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model_name = type(model).__name__.lower()

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# -------- Training Loop --------
best_val_acc = 0.0
train_losses, val_losses, train_accuracies,val_accuracies,all_labels, all_preds =train(train_loader=train_loader, val_loader=val_loader,optimizer=optimizer, criterion=criterion,scheduler=scheduler,best_val_acc=best_val_acc)
# -------- Graphs and Confusion Matrix --------
os.makedirs('./graphs', exist_ok=True)

# Plot: Epoch vs Loss
plt.figure()
plt.plot(range(1, 16), train_losses, label="Train Loss")
plt.plot(range(1, 16), val_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.savefig(f"./graphs/{model_name}_epoch_vs_loss.png")
plt.close()

# Plot: Epoch vs Accuracy
plt.figure()
plt.plot(range(1, 16), train_accuracies, label="Train Acc")
plt.plot(range(1, 16), val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Epoch vs Accuracy")
plt.legend()
plt.savefig(f"./graphs/{model_name}_epoch_vs_accuracy.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(custom_class_to_idx.keys()))
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(f"./graphs/{model_name}_confusion_matrix.png")
plt.close()

# (Optional) Print classification report
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=custom_class_to_idx.keys()))
