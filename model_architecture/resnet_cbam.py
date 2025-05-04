from torchvision import models
import torch
from model_architecture.CBAM import CBAM
import torch.nn as nn
import os

class ResNetCBAM(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=5, pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet model")

        # Inject CBAM after each layer
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.cbam1(x)

        x = self.model.layer2(x)
        x = self.cbam2(x)

        x = self.model.layer3(x)
        x = self.cbam3(x)

        x = self.model.layer4(x)
        x = self.cbam4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x


num_classes = 5  # Change based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCBAM(model_name='resnet50', num_classes=5).to(device)
model_name = type(model).__name__.lower()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


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
