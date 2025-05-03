import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models

from torchvision.models import inception_v3

model = inception_v3(pretrained=True, aux_logits=True)  # Important: keep aux_logits=True during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.fc = nn.Linear(model.fc.in_features, 5)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 5)  # Auxiliary head
model = model.to(device)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0


def train(train_loader, val_loader, optimizer, criterion, scheduler, best_val_acc):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    all_labels, all_preds = [], []

    for epoch in range(40):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Handle Inception model with aux_logits
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
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

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels.clear()
        all_preds.clear()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
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
            torch.save(model.state_dict(), 'best_dr_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies, all_labels, all_preds
