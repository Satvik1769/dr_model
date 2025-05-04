import os
import torch
import torch.nn as nn
from torchvision.models import densenet121
from torchvision import transforms

# Load pretrained DenseNet-121 and modify the final classification layer
model = densenet121(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 5)  # 5 output classes for DR
model = model.to(device)

model_name = type(model).__name__.lower()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

def train(train_loader, val_loader, optimizer, criterion, scheduler, best_val_acc):
    global model
    for epoch in range(40):
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
