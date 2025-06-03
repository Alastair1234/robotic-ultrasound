import torch
from torch.utils.data import DataLoader
from dataset_hf import create_hf_dataset
from model import DinoV2Regressor
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = create_hf_dataset('data/imageInfo.json', 'data/images')
dataset = dataset.train_test_split(test_size=0.1)

def collate_fn(batch):
    images = torch.stack([torch.tensor(example['image']).permute(2,0,1) for example in batch])
    labels = torch.stack([torch.tensor(example['label']) for example in batch])
    return images, labels

train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset['test'], batch_size=8, shuffle=False, collate_fn=collate_fn)

model = DinoV2Regressor().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), 'dinov2_regressor.pth')
