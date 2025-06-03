import torch
from torch.utils.data import DataLoader
from dataset_hf import create_hf_dataset
from model import DinoV2Regressor
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = create_hf_dataset('data/imageInfo.json', 'data/images')
dataset = dataset.train_test_split(test_size=0.1)

def collate_fn(batch):
    images = torch.stack([torch.tensor(example['image']).permute(2,0,1) for example in batch])
    labels = torch.stack([torch.tensor(example['label']) for example in batch])
    return images, labels

test_loader = DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=collate_fn)

model = DinoV2Regressor().to(device)
model.load_state_dict(torch.load('dinov2_regressor.pth'))
model.eval()

criterion = nn.MSELoss()
total_loss = 0.0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Test Loss: {avg_loss:.6f}")
