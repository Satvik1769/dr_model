import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

class AttentionModel(nn.Module):
    def __init__(self, base_model="inception_resnet_v2", num_classes=5):
        super(AttentionModel, self).__init__()

        # Load the pretrained model from timm
        self.base_model = timm.create_model(base_model, pretrained=True, features_only=True)
        pt_depth = self.base_model.feature_info[-1]['num_chs']  # Get last feature map depth

        # Batch Normalization layer
        self.bn_features = nn.BatchNorm2d(pt_depth)    

        # Attention mechanism layers
        self.attn1 = nn.Conv2d(pt_depth, 64, kernel_size=1, padding=0)
        self.attn2 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.attn3 = nn.Conv2d(16, 8, kernel_size=1, padding=0)
        self.attn4 = nn.Conv2d(8, 1, kernel_size=1, padding=0)  # No activation inside Conv2d

        # Weighting layer
        self.up_c2 = nn.Conv2d(1, pt_depth, kernel_size=1, padding=0, bias=False)
        with torch.no_grad():
            self.up_c2.weight.fill_(1.0)  # Equivalent to initializing weights as ones

        # Fully connected layers
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(pt_depth, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.base_model(x)[-1]  # Extract last feature map
        features = self.bn_features(features)

        # Attention mechanism
        attn = F.relu(self.attn1(features))
        attn = F.relu(self.attn2(attn))
        attn = F.relu(self.attn3(attn))
        attn = torch.sigmoid(self.attn4(attn))  # Apply sigmoid activation separately

        # Expand attention weights
        attn = self.up_c2(attn)

        # Apply attention mask
        mask_features = features * attn
        gap_features = F.adaptive_avg_pool2d(mask_features, (1, 1)).view(x.size(0), -1)
        gap_mask = F.adaptive_avg_pool2d(attn, (1, 1)).view(x.size(0), -1)

        # Rescale GAP output
        gap = gap_features / (gap_mask + 1e-6)

        # Fully connected layers
        gap = self.dropout(gap)
        gap = F.relu(self.fc1(gap))
        gap = self.dropout(gap)
        out = self.fc2(gap)

        return out

# Instantiate the model
num_classes = 5  # Change based on your dataset
model = AttentionModel(base_model="inception_resnet_v2", num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = type(model).__name__.lower()
model = model.to(device)

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
