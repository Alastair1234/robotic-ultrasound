import torch
from torch.utils.data import DataLoader
from dataset_hf import create_hf_dataset
from model import DinoV2Regressor
import torch.nn as nn
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = create_hf_dataset('data/imageInfo.json', 'data/images')
dataset = dataset.train_test_split(test_size=0.1)

# DataLoader with normalization handled in dataset
def collate_fn(batch):
    images = torch.stack([torch.tensor(example['image']).permute(2, 0, 1) for example in batch])
    labels = torch.stack([torch.tensor(example['label']) for example in batch])
    return images, labels

test_loader = DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=collate_fn)

# Load label normalization parameters
label_mean = torch.tensor(np.load('label_mean.npy')).to(device)
label_std = torch.tensor(np.load('label_std.npy')).to(device)

# Model setup
model = DinoV2Regressor().to(device)
model.load_state_dict(torch.load('dinov2_regressor.pth'))
model.eval()

criterion = nn.MSELoss()
total_loss = 0.0

all_predictions = []
all_ground_truth = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # De-normalize outputs and labels to original scale
        outputs_denorm = outputs * label_std + label_mean
        labels_denorm = labels * label_std + label_mean

        loss = criterion(outputs_denorm, labels_denorm)
        total_loss += loss.item()

        all_predictions.append(outputs_denorm.cpu().numpy())
        all_ground_truth.append(labels_denorm.cpu().numpy())

avg_loss = total_loss / len(test_loader)
print(f"\nTest Loss (denormalized): {avg_loss:.6f}")

# Optional: Detailed analysis
predictions_array = np.vstack(all_predictions)
ground_truth_array = np.vstack(all_ground_truth)

# Calculate Mean Absolute Error (MAE) for additional insight
mae = np.mean(np.abs(predictions_array - ground_truth_array), axis=0)
print("\nMean Absolute Error per label:")
label_names = ['Pos_X', 'Pos_Y', 'Pos_Z', 'Rot_X', 'Rot_Y', 'Rot_Z']
for name, error in zip(label_names, mae):
    print(f"{name}: {error:.4f}")
