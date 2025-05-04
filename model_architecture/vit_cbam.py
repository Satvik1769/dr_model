import torch
import torch.nn as nn
from timm import create_model
from model_architecture.CBAM import CBAM
import os
from torchvision import transforms



class ViT_CBAM(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=5, pretrained=True):
        super(ViT_CBAM, self).__init__()
        self.vit = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.cbam = CBAM(in_channels=self.vit.embed_dim)
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        # Extract CLS token and apply CBAM
        cls_feat = x[:, 0].unsqueeze(-1).unsqueeze(-1)  # reshape to (B, C, 1, 1)
        cls_feat = self.cbam(cls_feat).squeeze(-1).squeeze(-1)  # back to (B, C)
        return self.classifier(cls_feat)



num_classes = 5  # Change based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_CBAM(num_classes=5).to(device)
model_name = type(model).__name__.lower()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])


def train(train_loader, val_loader, optimizer, criterion, scheduler, best_val_acc):
    global model
    for epoch in range(15):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step()

        print(f"Epoch [{epoch+1}/40] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/{model_name}_dr_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies, all_labels, all_preds
